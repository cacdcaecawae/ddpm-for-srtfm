from pathlib import Path
from typing import Union

import cv2
import torch
from PIL import Image

from dataset import _build_transform
from ddpm_simple import DDPM


def add_noise(image_path: Union[str, Path],
              output_path: Union[str, Path],
              diffusion_steps: int = 1000,
              timestep: int = 300,
              image_size: int = 512,
              channels: int = 3,
              device: str = "cuda") -> None:
    """Add DDPM forward noise to a single image and save the result."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = _build_transform(image_size, channels)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    ddpm = DDPM(device=device, n_steps=diffusion_steps)
    timestep = max(0, min(diffusion_steps - 1, timestep))
    t = torch.tensor([timestep], device=device, dtype=torch.long)
    eps = torch.randn_like(tensor)
    noisy = ddpm.sample_forward(tensor, t, eps)

    noisy = noisy.squeeze(0).clamp(-1, 1)
    noisy = ((noisy + 1) / 2).permute(1, 2, 0).cpu().numpy()
    noisy = (noisy * 255).astype("uint8")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    default_input = Path("./data/TFM/hr/example.png")
    default_output = Path("work_dirs/noisy_example.png")
    if not default_input.exists():
        raise FileNotFoundError(
            f"Input image not found at {default_input}. Update the path or place a sample image there."
        )
    add_noise(default_input, default_output)
