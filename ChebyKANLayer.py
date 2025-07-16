import torch
import torch.nn as nn

import numpy as np
from numpy.polynomial.legendre import legvander
from numpy.polynomial.chebyshev import chebvander

import torch.nn as nn
import torch.nn.functional as F
# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.epsilon = 1e-7
        self.pre_mul = False
        self.post_mul = False
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # View and repeat input degree + 1 times
        b,c_in = x.shape
        if self.pre_mul:
            mul_1 = x[:,::2]
            mul_2 = x[:,1::2]
            mul_res = mul_1 * mul_2
            x = torch.concat([x[:,:x.shape[1]//2], mul_res])
        x = x.view((b, c_in, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = torch.tanh(x)
        x = torch.tanh(x)
        x = torch.acos(x)
        # x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon))
        # # Multiply by arange [0 .. degree]
        x = x* self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        if self.post_mul:
            mul_1 = y[:,::2]
            mul_2 = y[:,1::2]
            mul_res = mul_1 * mul_2
            y = torch.concat([y[:,:y.shape[1]//2], mul_res])
        return y

def legendre_chebyshev_vander(x, degree, basis_type='legendre'):
    if basis_type == 'legendre':
        return legvander(x, degree)
    elif basis_type == 'chebyshev':
        return chebvander(x, degree)
    else:
        raise ValueError("basis_type must be 'legendre' or 'chebyshev'")


def decompose_to_time_domain_2d(data, max_order=12, low=(0, 2), high=(7, 12), basis_type='legendre'):
    """
    输入:
        data: Tensor, shape (L, D) - L为时间长度，D为特征维度
        max_order: 多项式最高阶
        low: 低频保留的阶数范围 (from_idx, to_idx)
        high: 高频保留的阶数范围 (from_idx, to_idx)
        basis_type: 'legendre' or 'chebyshev'
    返回:
        f_low, f_high: 分别为低频、高频重构后的时域数据，shape为 (L, D)
    """
    L, D = data.shape
    device = data.device

    # Step 1: FFT over time dimension
    fft_data = torch.fft.fft(data, dim=0)  # (L, D)
    fft_real = fft_data.real
    fft_imag = fft_data.imag

    # Step 2: Compute basis (Legendre or Chebyshev)
    x = np.linspace(-1, 1, L)
    basis = legendre_chebyshev_vander(x, max_order, basis_type=basis_type)  # (L, order+1)
    basis_torch = torch.tensor(basis, dtype=torch.float32, device=device)   # (L, order+1)

    # Step 3: Compute pseudo-inverse
    basis_pinv = torch.linalg.pinv(basis_torch)  # (order+1, L)

    # Step 4: Project to basis
    coeff_real = torch.matmul(basis_pinv, fft_real)  # (order+1, D)
    coeff_imag = torch.matmul(basis_pinv, fft_imag)  # (order+1, D)

    # Step 5: Select and reconstruct low/high bands
    def reconstruct_band(coeff_real, coeff_imag, basis_torch, from_idx, to_idx):
        basis_sel = basis_torch[:, from_idx:to_idx + 1]  # (L, freq_range)
        coeff_real_sel = coeff_real[from_idx:to_idx + 1, :]  # (freq_range, D)
        coeff_imag_sel = coeff_imag[from_idx:to_idx + 1, :]

        recon_real = torch.matmul(basis_sel, coeff_real_sel)  # (L, D)
        recon_imag = torch.matmul(basis_sel, coeff_imag_sel)  # (L, D)
        return recon_real + 1j * recon_imag  # (L, D)

    low_freq = reconstruct_band(coeff_real, coeff_imag, basis_torch, *low)
    high_freq = reconstruct_band(coeff_real, coeff_imag, basis_torch, *high)

    # Step 6: Inverse FFT to time domain
    f_low = torch.fft.ifft(low_freq, dim=0).real  # (L, D)
    f_high = torch.fft.ifft(high_freq, dim=0).real  # (L, D)

    return f_low, f_high
