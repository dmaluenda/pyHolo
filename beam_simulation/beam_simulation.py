import os

import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift




fft = lambda field, size: fftshift(fft2(ifftshift(field), (size,size)))
ifft = lambda spectr, size: ifftshift(ifft2(fftshift(spectr), size))

kind = "radial"


def richards_wolf(E, NA, lamb, z=0, size=None):
    """ Implementation of Richards-Wolf diffraction integral of plane wave spectrum
        via fft2. The spectrum is assumed to be centered at the origin.
    """
    if size is None:
        assert E.shape[0] == E.shape[1], (f"Field at the entrance pupil must be square."
                                          f"{E.shape[0]} != {E.shape[1]}")
        size = E.shape[0]

    # _, _, nc = E.shape

    # Focused field
    Ef = np.zeros((size, size, 3), np.complex128)

    # Coordinates in the Gaussian-reference sphere
    _, _, coords = get_EP_coords(NA, lamb, size)

    # y, x = coords["y"], coords["x"]
    # r2 = coords["r2"]
    mask = coords["mask"]
    sinth = coords["sinth"]
    costh = coords["costh"]
    sqcosth = coords["sqcosth"]
    sinphi = coords["sinphi"]
    cosphi = coords["cosphi"]

    # Old code. It should be identical to the one in the get_EP_coords function
    # y, x = np.mgrid[-ny//2:ny//2, -nx//2:nx//2]
    # y = y/y.max()*L
    # x = x/x.max()*L
    #
    # r2 = x*x+y*y
    # sinth2 = r2/f/f
    # mask = sinth2 < NA*NA
    # sinth = np.sqrt(sinth2)*(mask)
    #
    # costh = np.sqrt(1-sinth2, where=mask)
    # costh *= mask
    # sqcosth = np.sqrt(costh)
    # phi = np.arctan2(y, x)
    # sinphi = np.sin(phi)
    # cosphi = np.cos(phi)

    # Multiplicació pels vectors unitaris...
    Ef[:, :, 0] = (mask*E[:, :, 0]*(sinphi*sinphi+cosphi*cosphi*costh) +
                   mask*E[:, :, 1]*sinphi*cosphi*(costh-1))
    Ef[:, :, 0][mask] /= sqcosth[mask]
    Ef[:, :, 1] = (mask*E[:, :, 1]*(sinphi*sinphi*costh+cosphi*cosphi) +
                   mask*E[:, :, 0]*sinphi*cosphi*(costh-1)*mask)
    Ef[:, :, 1][mask] /= sqcosth[mask]
    Ef[:, :, 2] = -sinth*(E[:, :, 0]*cosphi + E[:, :, 1]*sinphi)
    Ef[:, :, 2][mask] /= sqcosth[mask]

    # Propagació
    if z != 0:
        H = np.exp(-2j*np.pi*z)*mask
        Ef[:, :, 0] *= H
        Ef[:, :, 1] *= H
        Ef[:, :, 2] *= H
    Ef[:, :, 0] = fft(Ef[:, :, 0], size)
    Ef[:, :, 1] = fft(Ef[:, :, 1], size)
    Ef[:, :, 2] = fft(Ef[:, :, 2], size)
    return Ef

def get_z_component(Ex, Ey, pixel_size, size=None, lamb=520e-3, z=0):
    """ Getting the Z component of the field from the X and Y components
        by imposing the Gauss law of the electric field.
    """
    if size is None:
        assert Ex.shape[0] == Ex.shape[1], (f"Field at the entrance pupil must be square."
                                            f"{Ex.shape[0]} != {Ex.shape[1]}")
        size = Ex.shape[0]

    Ax = fft(Ex, size)
    Ay = fft(Ey, size)

    y, x = np.mgrid[-size//2:size//2, -size//2:size//2]

    alpha = x / x.max() / pixel_size * lamb * 0.5
    beta = y / y.max() / pixel_size * lamb * 0.5

    gamma = np.sqrt(1+0j-alpha*alpha-beta*beta)
    Az = (alpha*Ax+beta*Ay)/gamma


    # Propaguem si s'escau
    if z > 0:
        H = np.exp(-2j*np.pi*gamma*z)
        H[alpha*alpha+beta*beta >= 1] = 0
        Az *= H
        Ax *= H
        Ay *= H
        Ex = ifft(Ax, size)
        Ey = ifft(Ay, size)

    Ez = ifft(Az, size)

    return Ex, Ey, Ez



def get_EP_coords(NA, lamb, n):
    f = 5 / lamb
    Lf = 16
    L = n * f / 4 / Lf

    y, x = np.mgrid[-n // 2:n // 2, -n // 2:n // 2]
    y = y / y.max() * L
    x = x / x.max() * L

    r2 = x * x + y * y
    sinth2 = r2 / f / f
    mask = sinth2 < NA * NA

    sinth = np.sqrt(sinth2) * (mask)
    costh = np.sqrt(1 - sinth2, where=mask)
    costh *= mask
    sqcosth = np.sqrt(costh)
    theta = np.arcsin(sinth) * mask

    theta_0 = np.arcsin(NA)
    alpha_0 = np.cos(theta_0)
    alpha_bar = (alpha_0 + 1) * 0.5

    phi = np.arctan2(y, x)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    return L, f, {"x": x, "y": y, "r2": r2, "mask": mask,
                  "theta": theta, "sinth": sinth, "costh": costh, "sqcosth": sqcosth,
                  "phi": phi, "sinphi": sinphi, "cosphi": cosphi,
                  "theta_0":alpha_0, "alpha_0": alpha_0, "alpha_bar": alpha_bar}



def beam_sim_main():
    print('beam_sim_main')
    os.system('jupyter notebook beam_simulation/beam_sim_gui.ipynb')