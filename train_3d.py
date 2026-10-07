import json
import math
import random
from pathlib import Path
import torch

from gaussian import *
from utils import *
from densification_3d import density_control_3d

def load_training_cameras(root, device):
    root = Path(root)

    with open(root / "cameras.json") as file:
        data = json.load(file)

    K = torch.tensor(data["K"], dtype=torch.float32, device=device)

    def load_frames(frames):
        cameras = []
        for frame in frames:
            cam = {
                "R": torch.tensor(
                    frame["R_wc"],
                    dtype=torch.float32,
                    device=device,
                ),
                "t": torch.tensor(
                    frame["t"],
                    dtype=torch.float32,
                    device=device,
                ),
                "K": K,
                "H": data["height"],
                "W": data["width"],
                "image": load_image(root / frame["file"], device),
            }
            cameras.append(cam)
        return cameras

    train_cameras = load_frames(data["frames"])
    val_cameras = load_frames(data["val_frames"])
    return train_cameras, val_cameras


def initialize_3d(N, device):
    '''
    basically the same as 2d gaussian initialization
    '''
    quat = torch.zeros(N, 4, device=device)
    quat[:, 0] = 1.0

    g = {
        "mu": (torch.rand(N, 3, device=device) * 2 - 1) * 1.5,
        "log_s": torch.full((N, 3), math.log(0.08), device=device),
        "quat": quat,
        "color": torch.zeros(N, 3, device=device),
        "op_raw": torch.full((N,), -2.0, device=device),
    }

    return {
        name: torch.nn.Parameter(value)
        for name, value in g.items()
    }


def render_camera(g, cam):
    '''
    keep same black-background compositing as P2
    '''
    Sigma3 = covariance_3d(
        g["log_s"].exp(),
        g["quat"])

    mu_cam = g["mu"] @ cam["R"].T + cam["t"]
    visible = mu_cam[:, 2] > 1e-3

    if not visible.any():
        # A black image with a connection to the parameters.
        return (
            torch.zeros(
                cam["H"], cam["W"], 3,
                device=g["mu"].device,
                dtype=g["mu"].dtype,
            )
            + g["mu"].sum() * 0)

    mu2, Sigma2, depth = project_gaussian(
        g["mu"][visible],
        Sigma3[visible],
        cam["R"],
        cam["t"],
        cam["K"])

    # nearest, positive depth gaussians first 
    order = torch.argsort(depth, descending=False) 

    return render(
        mu2,
        Sigma2,
        g["color"][visible].sigmoid(),
        g["op_raw"][visible].sigmoid(),
        order,
        cam["H"],
        cam["W"])


def fit_3d(
    train_cameras,
    N=1024,
    steps=1500,
    seed=0,
    can_densify=False,
    budget=None,
    grad_threshold=2e-4,
    world_size_threshold=0.08,
):
    torch.manual_seed(seed)
    random.seed(seed)

    device = train_cameras[0]["K"].device
    budget = N if budget is None else budget

    if N > budget:
        raise ValueError("budget surpassed for gaussian count!")

    g = initialize_3d(N, device)
    opt = torch.optim.Adam(g.values(), lr=1e-2)

    grad_sum = torch.zeros(N, device=device)
    grad_steps = 0

    for step in range(steps):
        cam = random.choice(train_cameras)

        opt.zero_grad()

        image = render_camera(g, cam)
        loss = ((image - cam["image"]) ** 2).mean()
        loss.backward()

        if can_densify:
            with torch.no_grad():
                grad_sum += g["mu"].grad.norm(dim=-1)
            grad_steps += 1

        opt.step()

        if (
            can_densify
            and (step + 1) % 200 == 0
            and step + 1 < steps
        ):
            grad_mag = grad_sum / grad_steps

            print(
                "Gradient mean:", grad_mag.mean().item(),
                "Gradient max:", grad_mag.max().item(),
            )

            g, opt = density_control_3d(
                g,
                opt,
                grad_mag,
                budget=budget,
                grad_threshold=grad_threshold,
                size_threshold=world_size_threshold,
            )

            grad_sum = torch.zeros(len(g["mu"]), device=device)
            grad_steps = 0

        # if (step + 1) % 10 == 0:
        if (step + 1) % 100 == 0:
            score = -10 * torch.log10(
                loss.detach().clamp_min(1e-12)
            )
            print(step + 1, "Sampled-view PSNR:", score.item())

    return g


def evaluate_training(g, cameras, output_dir, show_ids=(0, 1, 2)):
    with torch.no_grad():
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scores = []

        for i, cam in enumerate(cameras):
            image = render_camera(g, cam)
            scores.append(psnr(image, cam["image"]))

            if i in show_ids:
                comparison = torch.cat([
                    cam["image"],
                    image,
                ], dim=1)

                save_image(
                    comparison,
                    output_dir / f"comparison_{i:03d}.png",
                )

        return sum(scores) / len(scores)

###################################################################################################

def run_p8():
    device = get_device()

    train_cameras, val_cameras = load_training_cameras(
        "spheres",
        device,
    )

    output_dir = Path("results/3d/p8")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load P7 instead of retraining it.
    state = torch.load(
        "results/3d/p7/model.pt",
        map_location=device,
        weights_only=True,
    )

    g_fixed = {
        name: value.to(device)
        for name, value in state.items()
    }

    # Use the actual P7 count as the P8 budget.
    budget = len(g_fixed["mu"])
    initial_count = max(1, budget // 4)

    # Match your actual P7 iteration count.
    steps = 1500

    grad_threshold = 2e-4
    world_size_threshold = 0.08

    g_dense = fit_3d(
        train_cameras,
        N=initial_count,
        steps=steps,
        can_densify=True,
        budget=budget,
        grad_threshold=grad_threshold,
        world_size_threshold=world_size_threshold,
    )

    # Save before evaluation.
    torch.save(
        {name: p.detach().cpu() for name, p in g_dense.items()},
        output_dir / "model.pt",
    )

    fixed_train = evaluate_training(
        g_fixed,
        train_cameras,
        output_dir / "fixed_train",
        show_ids=(0, 12, 24),
    )

    fixed_val = evaluate_training(
        g_fixed,
        val_cameras,
        output_dir / "fixed_val",
        show_ids=(0, 2, 5),
    )

    dense_train = evaluate_training(
        g_dense,
        train_cameras,
        output_dir / "dense_train",
        show_ids=(0, 12, 24),
    )

    dense_val = evaluate_training(
        g_dense,
        val_cameras,
        output_dir / "dense_val",
        show_ids=(0, 2, 5),
    )

    # Same held-out camera for both models.
    with torch.no_grad():
        cam = val_cameras[2]

        fixed_image = render_camera(g_fixed, cam)
        dense_image = render_camera(g_dense, cam)

        save_image(fixed_image, output_dir / "heldout_fixed.png")
        save_image(dense_image, output_dir / "heldout_dense.png")

        comparison = torch.cat([
            cam["image"],
            fixed_image,
            dense_image,
        ], dim=1)

        save_image(comparison, output_dir / "heldout_comparison.png")

    with open(output_dir / "metrics.txt", "w") as file:
        file.write(f"Steps: {steps}\n")
        file.write(f"Adaptive initial count: {initial_count}\n")
        file.write(f"Budget: {budget}\n")
        file.write(f"Gradient threshold: {grad_threshold}\n")
        file.write(f"World size threshold: {world_size_threshold}\n\n")

        file.write(
            f"Fixed: count={len(g_fixed['mu'])}, "
            f"train PSNR={fixed_train:.4f}, "
            f"held-out PSNR={fixed_val:.4f}\n"
        )

        file.write(
            f"Densified: count={len(g_dense['mu'])}, "
            f"train PSNR={dense_train:.4f}, "
            f"held-out PSNR={dense_val:.4f}\n"
        )

    print("Fixed:", len(g_fixed["mu"]), fixed_train, fixed_val)
    print("Densified:", len(g_dense["mu"]), dense_train, dense_val)


def run_p7(): 
    device = get_device()
    train_cameras, val_cameras = load_training_cameras("spheres", device)

    N = 4000
    steps = 1500

    output_dir = Path("results/3d/p7")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Training views:", len(train_cameras))

    g = fit_3d(
        train_cameras,
        N=N,
        steps=steps,
    )

    # Save before image output, so a saving error won't lose training.
    torch.save(
        {name: p.detach().cpu() for name, p in g.items()},
        output_dir / "model.pt",
    )

    mean_psnr = evaluate_training(
        g,
        train_cameras,
        output_dir,
    )

    with open(output_dir / "settings.txt", "w") as file:
        file.write(f"Gaussians: {N}\n")
        file.write(f"Steps: {steps}\n")
        file.write("Adam learning rate: 0.01\n")
        file.write(f"Mean training PSNR: {mean_psnr:.4f} dB\n")
        file.write("Centers initialized uniformly in [-1.5, 1.5]^3\n")
        file.write("Initial scale: 0.08 on each axis\n")
        file.write("Initial quaternion: (1, 0, 0, 0)\n")
        file.write("Initial RGB: 0.5\n")
        file.write("Initial opacity: approximately 0.119\n")

    print("Mean training PSNR:", mean_psnr)


if __name__ == "__main__":
    run_p8()