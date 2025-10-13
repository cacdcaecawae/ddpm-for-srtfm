import sys
from pathlib import Path

# Allow running tests without installing the project as a package.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from ddpm_simple import DDPM
from ddim import DDIM


class DummyNet(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class DummySRNet(torch.nn.Module):
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                lr: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_ddpm_backward_shape():
    device = torch.device("cpu")
    scheduler = DDPM(device, n_steps=10)
    net = DummyNet()
    output = scheduler.sample_backward((2, 3, 8, 8), net, device=device)
    assert output.shape == (2, 3, 8, 8)


def test_ddpm_sr_backward_shape():
    device = torch.device("cpu")
    scheduler = DDPM(device, n_steps=10)
    net = DummySRNet()
    lr = torch.randn(2, 3, 8, 8)
    output = scheduler.sample_backward_sr((2, 3, 8, 8),
                                          net,
                                          lr,
                                          device=device)
    assert output.shape == (2, 3, 8, 8)


def test_ddim_backward_matches_shape():
    device = torch.device("cpu")
    scheduler = DDIM(device, n_steps=20)
    net = DummySRNet()
    lr = torch.randn(1, 3, 8, 8)
    output = scheduler.sample_backward_sr((1, 3, 8, 8),
                                          net,
                                          lr,
                                          device=device,
                                          ddim_step=5,
                                          eta=0.0)
    assert output.shape == (1, 3, 8, 8)
