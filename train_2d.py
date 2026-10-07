import csv
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from utils import * 
from gaussian import * 

def initialize_2d(N, H, W, device):
    '''
    store initialized gaussian parameters in dict g --> contains params for all N gaussians 
    '''
    g = {
        "mu": torch.rand(N, 2, device=device) # spread out blobs
              * torch.tensor([W, H], device=device),

        "log_s": torch.full( # scale is optimized in log space
            (N, 2),
            fill_value=float(torch.log(torch.tensor(0.02 * max(H, W)))),
            device=device,
        ),

        "theta": torch.zeros(N, device=device),
        "color": torch.zeros(N, 3, device=device),
        "op_raw": torch.full((N,), -2.0, device=device), # opacity value before sigmoid
    }

    # convert tensors to learnable parameters 
    for name, value in g.items():
        g[name] = torch.nn.Parameter(value)

    return g


def render_2d(g, H, W):
    '''
    alpha-composite onto 2d image, return rgb color of HxW pixels 
    '''
    Sigma = covariance_2d(g["log_s"].exp(), g["theta"])

    N = len(g["mu"])
    order = torch.arange(N, device=g["mu"].device)
    return render(g["mu"], Sigma, g["color"].sigmoid(), g["op_raw"].sigmoid(), order, H, W)


def fit_2d(target, N, steps=2000, seed=0, can_densify=False, budget=None,):
    ''' 
    learn optimal gaussian parameters (g) to make gaussian blobs reproduce the target image, 
    also returns final rendered img and psnr 
    '''
    torch.manual_seed(seed) # random initializations of blob positions made repeatable

    H, W = target.shape[:2]

    if budget is None: 
        budget = N

    if N > budget:
        raise ValueError("initial # of gaussians exceeds budget")

    g = initialize_2d(N, H, W, target.device)
    opt = torch.optim.Adam(g.values(), lr=1e-2)

    grad_sum = torch.zeros(N, device=target.device)
    grad_steps = 0

    for step in range(steps):
        opt.zero_grad()

        image = render_2d(g, H, W)
        loss = ((image - target) ** 2).mean()

        loss.backward()

        # accumulate total grad for each gaussian over each step --> later score them by the mean magnitude of the gradient of the loss w/r to its position
        if can_densify:
            with torch.no_grad():
                grad_sum += g["mu"].grad.norm(dim=-1)
            grad_steps += 1

        opt.step()

        # DENSIFICATION PASS every few hundred steps besides the last one:
        if can_densify and (step + 1) % 200 == 0 and step + 1 < steps:
            old_count = len(g["mu"])
            g, opt = density_control(g, opt, grad_mag=grad_sum / grad_steps, budget=budget, image_width=W)

            print(
                f"step {step + 1}: "
                f"{old_count} -> {len(g['mu'])} gaussians"
            )

            # restart grad accumulation so that it only tracks steps since last densification pass
            grad_sum = torch.zeros(len(g["mu"]), device=target.device)
            grad_steps = 0

        # keep track of psnr 
        if (step + 1) % 500 == 0:
            score = -10 * torch.log10(loss.detach().clamp_min(1e-12))
            print(step + 1, "PSNR:", score.item())

    # render post-training image
    with torch.no_grad():
        image = render_2d(g, H, W)
        score = psnr(image, target)

    return g, image, score

#########################################################################################################

def run_p5():
    '''
    densification comparisons: fit each target independently at several counts
    '''
    device = get_device()

    # counts = [256, 1024, 4096]
    counts = [256, 512, 1024] # less time...
    names = ["coffee", "astronaut", "cat"]

    output_dir = Path("results/2d/p5")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        target = load_image(
            f"data/{name}.png",
            device,
            size=(128, 128),)

        save_image(target, output_dir / f"{name}_target.png")

        for N in counts:
            g, image, score = fit_2d(
                target,
                N=N,
                steps=2000,
                seed=0,
                can_densify=False)

            # Save the learned parameters before saving images.
            torch.save(
                {key: value.detach().cpu() for key, value in g.items()},
                output_dir / f"{name}_N{N}.pt")

            save_image(image, output_dir / f"{name}_N{N}.png")

            comparison = torch.cat([target, image], dim=1)
            save_image(
                comparison,
                output_dir / f"{name}_N{N}_comparison.png")

            rows.append({
                "image": name,
                "N": N,
                "PSNR": score,
            })

            # Update the results file after each completed experiment.
            with open(output_dir / "psnr.csv", "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["image", "N", "PSNR"])
                writer.writeheader()
                writer.writerows(rows)

    fig, ax = plt.subplots()
    for name in names:
        results = [row for row in rows if row["image"] == name]
        ax.plot(
            [row["N"] for row in results],
            [row["PSNR"] for row in results],
            marker="o",
            label=name)

    ax.set_xlabel("Number of Gaussians")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "psnr_vs_count.png")
    plt.close(fig)



if __name__ == "__main__":
    run_p5()

    
    # device = get_device()
    # target = load_image("data/coffee.png", device, size=(128, 128))

    # # Start small and grow toward the budget.
    # g_dense, image_dense, score_dense = fit_2d(
    #     target,
    #     N=256,
    #     can_densify=True,
    #     budget=1024,
    # )

    # # The instructions require the same FINAL count for the comparison.
    # final_count = len(g_dense["mu"])

    # g_fixed, image_fixed, score_fixed = fit_2d(
    #     target,
    #     N=final_count,
    #     can_densify=False,
    # )

    # save_image(target, "results/2d/p4_target.png")
    # save_image(image_dense, "results/2d/p4_densified.png")
    # save_image(image_fixed, "results/2d/p4_fixed.png")

    # comparison = torch.cat([
    #     target,
    #     image_fixed,
    #     image_dense,
    # ], dim=1)

    # save_image(comparison, "results/2d/p4_comparison.png")

    # save_model(g_dense, "results/2d/p4_densified.pt")
    # save_model(g_fixed, "results/2d/p4_fixed.pt")

    # print("Final Gaussian count:", final_count)
    # print("Fixed PSNR:", score_fixed)
    # print("Densified PSNR:", score_dense)
    # print("PSNR difference:", score_dense - score_fixed)