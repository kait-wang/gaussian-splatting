import math
import torch

def density_control(
    g,
    opt,
    grad_mag,
    budget,
    image_width,
    grad_threshold=1e-5,
    size_threshold=0.02,
    split_scale=1.6,
    prune_opacity=0.005,
):
    '''
    adaptive density control - clone/split/prune gaussians in high-error areas to enable higher/lower detail 
    '''
    with torch.no_grad():
        scale = g["log_s"].exp()
        opacity = g["op_raw"].sigmoid()

        # 1. PRUNE nearly transparent gaussians 
        alive = opacity >= prune_opacity # make mask for pruning 

        if not alive.any():
            raise RuntimeError("prune opacity threshold too high, no gaussians would be left.")

        candidates = torch.where(alive & (grad_mag > grad_threshold))[0]
        candidates = candidates[torch.argsort(grad_mag[candidates], descending=True)] # prioritize gaussians with high gradient (need more adjustment)

        # update densification budget 
        room = max(0, budget - int(alive.sum().item()))
        selected = candidates[:room]

        print(
        "budget:", budget,
        "room:", room,
        "gradient mean:", grad_mag.mean().item(),
        "gradient max:", grad_mag.max().item(),
        "above threshold:", len(candidates),
        "pruned:", (~alive).sum().item(),
        )

        # 2. CLONE small gaussians, SPLIT big gaussians 
        small = scale.amax(dim=-1) <= size_threshold * image_width

        clone_ids = selected[small[selected]]
        split_ids = selected[~small[selected]]

        # 3. remove original gaussians that were split, keep the ones that were cloned
        keep = alive.clone()
        keep[split_ids] = False
        keep_ids = torch.where(keep)[0]

        # keep track of which old gaussians contribute to initial vals of new gaussians 
        sources = torch.cat([keep_ids, clone_ids, split_ids.repeat_interleave(2),]) 
        values = {
            name: parameter.detach()[sources].clone()
            for name, parameter in g.items()
        }

        n_keep = len(keep_ids)
        child_start = n_keep + len(clone_ids)

        # move and shrink the post-split gaussians (children)
        if len(split_ids) > 0:
            parent_mu = g["mu"][split_ids]
            parent_scale = scale[split_ids]
            theta = g["theta"][split_ids]

            local_offset = torch.randn_like(parent_scale) * parent_scale

            c = torch.cos(theta)
            s = torch.sin(theta)

            dx = c * local_offset[:, 0] - s * local_offset[:, 1]
            dy = s * local_offset[:, 0] + c * local_offset[:, 1]
            offset = torch.stack([dx, dy], dim=-1)

            # children positioned on each side of parent's center, orientation matched  
            children_mu = torch.stack([
                parent_mu + offset,
                parent_mu - offset,
            ], dim=1).reshape(-1, 2)

            values["mu"][child_start:] = children_mu
            values["log_s"][child_start:] -= math.log(split_scale)

        # make new gaussians learnable params through Adam optimizer!!
        new_g = {
            name: torch.nn.Parameter(value)
            for name, value in values.items()
        }

        new_opt = torch.optim.Adam(new_g.values(), lr=opt.param_groups[0]["lr"])

        return new_g, new_opt
