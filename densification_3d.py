# same as densification for 2D, but 

import math
import torch

from gaussian import quaternion_to_rotation

@torch.no_grad()
def density_control_3d(
    g,
    opt,
    grad_mag,
    budget,
    grad_threshold=2e-4,
    size_threshold=0.08,
    split_scale=1.6,
    prune_opacity=0.005,
):
    scale = g["log_s"].exp()
    opacity = g["op_raw"].sigmoid()

    # Prune nearly transparent Gaussians.
    alive = opacity >= prune_opacity

    if not alive.any():
        raise RuntimeError("Pruning would remove every Gaussian.")

    # Select high-gradient Gaussians, highest scores first.
    candidates = torch.where(
        alive & (grad_mag > grad_threshold)
    )[0]

    candidates = candidates[
        torch.argsort(grad_mag[candidates], descending=True)
    ]

    # Each clone or two-child split increases count by one.
    room = max(0, budget - int(alive.sum().item()))
    selected = candidates[:room]

    # This threshold is in world units, not pixels.
    small = scale.amax(dim=-1) <= size_threshold

    clone_ids = selected[small[selected]]
    split_ids = selected[~small[selected]]

    # Keep clone parents; replace split parents.
    keep = alive.clone()
    keep[split_ids] = False
    keep_ids = torch.where(keep)[0]

    # Copy all parameters into the new Gaussian set.
    sources = torch.cat([
        keep_ids,
        clone_ids,
        split_ids.repeat_interleave(2),
    ])

    values = {
        name: parameter.detach()[sources].clone()
        for name, parameter in g.items()
    }

    child_start = len(keep_ids) + len(clone_ids)

    if len(split_ids) > 0:
        parent_mu = g["mu"][split_ids]
        parent_scale = scale[split_ids]

        # Sample offsets along the parent's three local axes.
        local_offset = torch.randn_like(parent_scale) * parent_scale

        # Rotate offsets into world coordinates.
        R = quaternion_to_rotation(g["quat"][split_ids])
        offset = (
            R @ local_offset.unsqueeze(-1)
        ).squeeze(-1)

        children_mu = torch.stack([
            parent_mu + offset,
            parent_mu - offset,
        ], dim=1).reshape(-1, 3)

        values["mu"][child_start:] = children_mu
        values["log_s"][child_start:] -= math.log(split_scale)

    new_g = {
        name: torch.nn.Parameter(value)
        for name, value in values.items()
    }

    # Restart Adam, following the instructor clarification.
    new_opt = torch.optim.Adam(
        new_g.values(),
        lr=opt.param_groups[0]["lr"],
    )

    print(
        "Pruned:", int((~alive).sum().item()),
        "Eligible:", len(candidates),
        "Cloned:", len(clone_ids),
        "Split parents:", len(split_ids),
        "Count:", len(g["mu"]), "->", len(new_g["mu"]),
    )

    return new_g, new_opt