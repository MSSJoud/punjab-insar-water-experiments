from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass(frozen=True)
class PhysicsConfig:
    E: float = 1e9
    nu: float = 0.25
    rho_w: float = 1000.0
    g: float = 9.81
    alpha: float = 0.8
    Hg: float = 150.0
    Seff: float = 0.2
    dx: float = 10000.0
    dy: float = 10000.0
    a_load: float = 3000.0
    a_poro: float = 3000.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_elastic_kernel(E: float, nu: float, dx: float, dy: float, a: float, nx: int, ny: int) -> np.ndarray:
    xgrid = (np.arange(nx) - nx / 2) * dx
    ygrid = (np.arange(ny) - ny / 2) * dy
    xx, yy = np.meshgrid(xgrid, ygrid)
    r = np.sqrt(xx ** 2 + yy ** 2)
    r[r < 1e-6] = 1e-6
    return (1 + nu) / (np.pi * E * (1 - nu)) * (1 - np.exp(-r / a)) / r


def build_poroelastic_kernel(
    E: float,
    nu: float,
    alpha: float,
    hg: float,
    dx: float,
    dy: float,
    a: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    xgrid = (np.arange(nx) - nx / 2) * dx
    ygrid = (np.arange(ny) - ny / 2) * dy
    xx, yy = np.meshgrid(xgrid, ygrid)
    r = np.sqrt(xx ** 2 + yy ** 2)
    r[r < 1e-6] = 1e-6
    factor = alpha * (1 + nu) * hg * 9.81 / (np.pi * E * (1 - nu))
    return factor * (1 - np.exp(-r / a)) / r


def build_fft_kernels(ny: int, nx: int, physics: PhysicsConfig, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g_load = build_elastic_kernel(physics.E, physics.nu, physics.dx, physics.dy, physics.a_load, nx, ny)
    g_poro = build_poroelastic_kernel(
        physics.E,
        physics.nu,
        physics.alpha,
        physics.Hg,
        physics.dx,
        physics.dy,
        physics.a_poro,
        nx,
        ny,
    )
    g_load_fft = torch.fft.fft2(torch.fft.ifftshift(torch.tensor(g_load, dtype=torch.float32, device=device)))
    g_poro_fft = torch.fft.fft2(torch.fft.ifftshift(torch.tensor(g_poro, dtype=torch.float32, device=device)))
    return g_load_fft, g_poro_fft


def fft_convolve2d(field: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.fft2(field) * kernel_fft).real


def forward_physics_torch(
    y_pred: torch.Tensor,
    g_load_fft: torch.Tensor,
    g_poro_fft: torch.Tensor,
    physics: PhysicsConfig,
    sg_index: int = 3,
) -> torch.Tensor:
    delta_l = physics.rho_w * y_pred[:, :3].sum(dim=1)
    delta_p = physics.rho_w * physics.g * (y_pred[:, sg_index] / physics.Seff)
    uz_load = fft_convolve2d(delta_l, g_load_fft)
    uz_poro = fft_convolve2d(delta_p, g_poro_fft)
    return (uz_load + uz_poro).unsqueeze(1)


def forward_two_layer_torch(
    y_pred: torch.Tensor,
    g_load_fft: torch.Tensor,
    g_poro_fft: torch.Tensor,
    physics: PhysicsConfig,
) -> torch.Tensor:
    delta_l = physics.rho_w * y_pred[:, 0]
    delta_p = physics.rho_w * physics.g * (y_pred[:, 1] / physics.Seff)
    uz_load = fft_convolve2d(delta_l, g_load_fft)
    uz_poro = fft_convolve2d(delta_p, g_poro_fft)
    return (uz_load + uz_poro).unsqueeze(1)
