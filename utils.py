import torch 
import numpy as np 
from PIL import Image
from pathlib import Path

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def psnr(image, target):
    mse = ((image - target) ** 2).mean()
    return (-10 * torch.log10(mse.clamp_min(1e-12))).item()

def load_image(path, device, size=None):
    image = Image.open(path).convert("RGB")

    if size is not None:
        image = image.resize(size, Image.Resampling.LANCZOS)

    array = np.array(image, dtype=np.float32) / 255.0
    return torch.tensor(array, device=device)


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().clamp(0, 1).cpu().numpy()
    array = np.rint(array * 255).astype(np.uint8)
    Image.fromarray(array).save(path)

def save_model(g, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: p.detach().cpu() for name, p in g.items()}, path)