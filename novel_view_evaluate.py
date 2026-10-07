import math
from pathlib import Path

import torch

from train_3d import (
    load_training_cameras,
    render_camera,
    evaluate_training,
)

from utils import get_device, save_image

def make_novel_camera(cam, angle_degrees=7.0):
    '''
    just rotates an existing camera around the world origin
    '''
    angle = math.radians(angle_degrees)

    c = math.cos(angle)
    s = math.sin(angle)

    Q = torch.tensor([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], device=cam["R"].device, dtype=cam["R"].dtype)

    # Camera center in world coordinates.
    center = -cam["R"].T @ cam["t"]

    # Rotate the camera pose around the world origin.
    new_center = Q @ center
    new_R = cam["R"] @ Q.T
    new_t = -new_R @ new_center

    return {
        "R": new_R,
        "t": new_t,
        "K": cam["K"],
        "H": cam["H"],
        "W": cam["W"],
    }


def main():
    device = get_device()

    train_cameras, val_cameras = load_training_cameras(
        "spheres",
        device,
    )

    # Use the completed P7 model.
    checkpoint = "results/3d/p7/model.pt"

    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    g = {
        name: value.to(device)
        for name, value in state.items()
    }

    output_dir = Path("results/3d/p9")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate the final model across all training views.
    train_psnr = evaluate_training(
        g,
        train_cameras,
        output_dir / "train",
        show_ids=(),
    )

    # Evaluate all 12 held-out views and save example comparisons.
    heldout_psnr = evaluate_training(
        g,
        val_cameras,
        output_dir / "heldout",
        show_ids=(0, 2, 5),
    )

    # Render three additional viewpoints.
    # The assignment allows a few frames instead of a video.
    with torch.no_grad():
        for i, angle in enumerate([-7.0, 7.0, 13.0]):
            novel_cam = make_novel_camera(
                train_cameras[0],
                angle_degrees=angle,
            )

            image = render_camera(g, novel_cam)

            save_image(
                image,
                output_dir / f"novel_{i:03d}.png",
            )

    gap = train_psnr - heldout_psnr

    with open(output_dir / "metrics.txt", "w") as file:
        file.write(f"Checkpoint: {checkpoint}\n")
        file.write(f"Gaussian count: {len(g['mu'])}\n")
        file.write(f"Mean training PSNR: {train_psnr:.4f} dB\n")
        file.write(f"Mean held-out PSNR: {heldout_psnr:.4f} dB\n")
        file.write(f"Training minus held-out PSNR: {gap:.4f} dB\n")

    print("Gaussians:", len(g["mu"]))
    print("Mean training PSNR:", train_psnr)
    print("Mean held-out PSNR:", heldout_psnr)
    print("Training minus held-out PSNR:", gap)


if __name__ == "__main__":
    main()