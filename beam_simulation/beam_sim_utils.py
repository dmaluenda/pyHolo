import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
# from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from numpy.fft import fft2, ifft2, fftshift, ifftshift


# Hi there

def get_EP_coords(size, res_PE, lamb=1):

    # pixel_size = lamb / 20  # in focal plane
    # width_EP = size * lamb / pixel_size  # in Entrance Pupil

    y, x = np.mgrid[-size // 2:size // 2, -size // 2:size // 2]
    y = y.astype(np.float64) / res_PE
    x = x.astype(np.float64) / res_PE
    # y = y / y.max() * width_EP / 2
    # x = x / x.max() * width_EP / 2

    rho2 = x * x + y * y

    res_EP = np.abs(np.argmin(np.abs(rho2-1), axis=0).min()-size//2)  # maximum radius of the EP. It will correspond to theta=pi/2

    sinth2 = rho2 # / rho0 / rho0  # sin(theta)^2. It gets 1 in the edge of the EP
    premask = sinth2 < 1  # mask for the EP. Outside it makes no sense

    sinth = np.sqrt(sinth2, where=premask)
    costh = np.sqrt(1 - sinth2, where=premask)
    costh *= premask
    sqcosth = np.sqrt(costh, where=premask)
    theta = np.arcsin(sinth, where=premask)

    # theta_0 = np.arcsin(NA/refractive_index)
    # alpha_0 = np.cos(theta_0)
    # alpha_bar = (alpha_0 + 1) * 0.5

    phi = np.arctan2(y, x)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    return {"x": x, "y": y, "r2": rho2, "res_EP": res_EP, "premask": premask,
            "theta": theta, "sinth": sinth, "costh": costh, "sqcosth": sqcosth,
            "phi": phi, "sinphi": sinphi, "cosphi": cosphi}
            # "pixel_size": pixel_size, "width_EP": width_EP}
            # "theta_0":alpha_0, "alpha_0": alpha_0, "alpha_bar": alpha_bar}




def richards_wolf(entrance_beam, coord_EP, NA, lamb, z=0, size_ratio=1, refractive_index=1):
    """ Implementation of Richards-Wolf diffraction integral of plane wave spectrum
        via fft2. The spectrum is assumed to be centered at the origin.
    """

    fft = lambda field, size: fftshift(fft2(ifftshift(field), (size, size)))
    ifft = lambda spectr, size: ifftshift(ifft2(fftshift(spectr), size))

    assert entrance_beam.shape[0] == entrance_beam.shape[1], (
            f"Field at the entrance pupil must be square."
            f"{entrance_beam.shape[0]} != {entrance_beam.shape[1]}")

    size = int(entrance_beam.shape[0] * size_ratio)

    # _, _, nc = entrance_beam.shape

    # Coordinates in the Gaussian-reference sphere
    premask = coord_EP["premask"]
    sinth = coord_EP["sinth"]
    costh = coord_EP["costh"]
    sqcosth = coord_EP["sqcosth"]
    sinphi = coord_EP["sinphi"]
    cosphi = coord_EP["cosphi"]

    mask = sinth < NA / refractive_index  # mask for the EP. Outside, you are out of NA
    mask *= premask

    plt.imshow(mask)
    plt.colorbar()
    plt.show()
    plt.figure()

    # Multiplicació pels vectors unitaris...  (original size, but 3D)
    bended_beam = np.zeros((entrance_beam.shape[0], entrance_beam.shape[1], 3),
                           dtype=np.complex128)

    bended_beam[:, :, 0] = (mask*entrance_beam[:, :, 0]*(sinphi*sinphi + cosphi*cosphi*costh)
                           +mask*entrance_beam[:, :, 1]*sinphi*cosphi*(costh-1))

    bended_beam[:, :, 1] = (mask*entrance_beam[:, :, 1]*(sinphi*sinphi*costh + cosphi*cosphi)
                           +mask*entrance_beam[:, :, 0]*sinphi*cosphi*(costh-1))

    bended_beam[:, :, 2] = -sinth*(entrance_beam[:, :, 0]*cosphi +
                                   entrance_beam[:, :, 1]*sinphi)

    # Apodization
    bended_beam[:, :, 0][mask] /= sqcosth[mask]
    bended_beam[:, :, 1][mask] /= sqcosth[mask]
    bended_beam[:, :, 2][mask] /= sqcosth[mask]

    # Propagació
    if z != 0:
        H = np.exp(-2j*np.pi*z)*mask
        bended_beam[:, :, 0] *= H
        bended_beam[:, :, 1] *= H
        bended_beam[:, :, 2] *= H

    # Focused field (full resolution)
    focused_beam = np.zeros((size, size, 3), np.complex128)
    focused_beam[:, :, 0] = fft(bended_beam[:, :, 0], size)
    focused_beam[:, :, 1] = fft(bended_beam[:, :, 1], size)
    focused_beam[:, :, 2] = fft(bended_beam[:, :, 2], size)

    return focused_beam


def plot_entrance_beam(beam, label='', fig_num=0, cmap_abs='jet', cmap_ph='twilight_shifted'):
    """ Plot the beam in the entrance pupil.
    """
    Ex = beam[:, :, 0]
    Ey = beam[:, :, 1]
    beam_max = np.abs(beam.max())

    fig = plt.figure(figsize=(10, 10))
    ax = ImageGrid(fig, 111,
                   nrows_ncols=(2, 3),
                   axes_pad=0.6,
                   cbar_location="right",
                   cbar_mode="edge",
                   cbar_size="5%",
                   cbar_pad=0.2
                   )
    intensity = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    ax[0].imshow(intensity/intensity.max(), vmin=0, vmax=1, cmap='gray')
    ax[0].set_title(r"$I = |Ex|^2 + |Ey|^2$")
    ax[1].imshow(np.abs(Ex)/beam_max, vmin=0, vmax=1, cmap=cmap_abs)
    ax[1].set_title(r"$|Ex|$")
    im1 = ax[2].imshow(np.abs(Ey)/beam_max, vmin=0, vmax=1, cmap=cmap_abs)
    ax[2].set_title(r"$|Ey|$")
    ax[3].imshow(np.angle(Ey/Ex), vmin=-np.pi, vmax=np.pi, cmap=cmap_ph)
    ax[3].set_title(r"Relative phase $= \phi_y - \phi_x$")
    ax[4].imshow(np.angle(Ex), vmin=-np.pi, vmax=np.pi, cmap=cmap_ph)
    ax[4].set_title(r"$\phi_x$")
    im2 = ax[5].imshow(np.angle(Ey), vmin=-np.pi, vmax=np.pi, cmap=cmap_ph)
    ax[5].set_title(r"$\phi_y$")
    cbar1 = fig.colorbar(im1, ax=ax, cax=ax.cbar_axes[0], orientation='vertical',
                         shrink=0.5, ticks=[0, 1])
    cbar1.ax.set_yticklabels([r'0', r'max'], fontsize=20)
    cbar2 = fig.colorbar(im2, ax=ax, cax=ax.cbar_axes[1], orientation='vertical',
                         ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar2.ax.set_yticklabels(
        [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=20)
    plt.show()
    title = f"({label} Pol.) The raw data of the retrieved transversal component."
    return print_fig(title, fig_num)

def plot_focused_field(focused_field, fig_num=0, trim=None, label="", verbose=1,
                       pixel_size=1, lamb=520e-3, ticks_step=1):
    """ Plot the focused field.
    """
    Ex = focused_field[:, :, 0]
    Ey = focused_field[:, :, 1]
    Ez = focused_field[:, :, 2]

    ntrim = -trim if trim else None
    lims = Ex[trim:ntrim, trim:ntrim].shape

    extension = lims[0] * pixel_size / lamb  # window size in lambdas
    half_side = extension / 2
    extent = [-half_side, half_side, -half_side, half_side]

    ticks = [0]
    for tt in range(int(extension / ticks_step / 2)):
        tt1 = tt + 1
        ticks.append(tt1 * ticks_step)
        ticks.insert(0, -tt1 * ticks_step)

    fig = plt.figure(figsize=(20, 40))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2, 5),
                    axes_pad=0.6,
                    cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="5%",
                    cbar_pad=0.2
                    )

    Ax = np.abs(Ex[trim:ntrim, trim:ntrim])
    Ay = np.abs(Ey[trim:ntrim, trim:ntrim])
    Az = np.abs(Ez[trim:ntrim, trim:ntrim])
    phx = np.angle(Ex[trim:ntrim, trim:ntrim])  # range [-pi, pi]
    phy = np.angle(Ey[trim:ntrim, trim:ntrim])  # range [-pi, pi]
    phz = np.angle(Ez[trim:ntrim, trim:ntrim])  # range [-pi, pi]

    IT = np.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2)
    Itrans = np.sqrt(Ax ** 2 + Ay ** 2)
    maxIT = np.max(IT)

    ph = np.angle(Ey/Ex)  # range [-pi, pi]
    # trans_stokes = None
    # exp_phase_shift = (np.arctan2(trans_stokes[3], trans_stokes[2]) if trans_stokes else
    #                    np.angle(Ey))  # range [-pi, pi]
    # exp_phase_shift = exp_phase_shift[trim:ntrim, trim:ntrim]
    # ph_error = ph_range(ph - exp_phase_shift) # range [-pi, pi]

    my_greens = np.zeros((256, 4))
    my_greens[:, 1] = np.linspace(0, 1, 256)
    my_greens[:, 3] = 1
    cmap_amps = ListedColormap(my_greens)
    cmap_phs = 'twilight_shifted'

    normalize = lambda a: a / a.max()  # (a-a.min())/(a.max()-a.min())

    if verbose > 1:
        get_title = lambda name, im: (fr'${name} ; '
                                      fr'vmin={im.min():.2g} ; '
                                      fr'vmax={im.max():.2g}$')
        fs = 14
    else:
        bra = '{'
        ket = '}'
        get_title = lambda name, im: (f'${name}$  $_{bra}[max={(im.max()):0.2f}]{ket}$')
        fs = 20

    for idx, ax in enumerate(axs):
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \, / \, \lambda$', fontsize=20)
        ax.set_ylabel(r'$y \, / \, \lambda$', fontsize=20)

        if idx == 0:
            im = ax.imshow(normalize(Ax), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title('|E_x|', Ax/maxIT), fontsize=fs)
        elif idx == 1:
            im = ax.imshow(normalize(Ay), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title('|E_y|', Ay/maxIT), fontsize=fs)
        elif idx == 2:
            im = ax.imshow(normalize(Itrans), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title(r'\sqrt{|E_x|^2+|E_y|^2}', Itrans/maxIT), fontsize=fs)
        elif idx == 3:
            im = ax.imshow(normalize(Az), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title('|E_z|', Az/maxIT), fontsize=fs)
        elif idx == 4:
            im = ax.imshow(normalize(IT), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title(r'\sqrt{I_T}', IT/maxIT), fontsize=fs)
        elif idx == 5:
            im = ax.imshow(phx, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta_x$', fontsize=fs)
        elif idx == 6:
            im = ax.imshow(phy, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta_y$', fontsize=fs)
        elif idx == 7:
            im = ax.imshow(ph, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\tilde{\delta}=\delta_y-\delta_x$', fontsize=fs)
        elif idx == 8:
            im = ax.imshow(phz, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta_z$', fontsize=fs)
        elif idx == 9:
            im = ax.imshow(np.zeros_like(Ax), cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            # ax.set_title(r'$\epsilon_\delta=\tilde{\delta}-\delta_{exp}$', fontsize=fs)


        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=20)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=20)
        if idx == 5:
            cbar = axs.cbar_axes[1].colorbar(im,  # FIXME: The line below is not working
            # cbar = fig.colorbar(im, ax=ax, cax=axs.cbar_axes[idx], orientation='vertical', shrink=0.5,
                                             ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2,
                                                    np.pi])
            cbar.ax.set_yticklabels(
                [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=20)
        elif idx == 2:
            cbar = fig.colorbar(im, ax=ax, cax=axs.cbar_axes[0], orientation='vertical',
                                shrink=0.5,
                                ticks=[0, 1])
            cbar.ax.set_yticklabels([r'0', r'max'], fontsize=20)

    plt.show()
    return print_fig(f"({label}) Field in the focal plane.", fig_num)

def print_fig(msg, fig_num):
    fig_num += 1
    print(f"Figure {fig_num}: {msg}")
    return fig_num


def beam_sim_main(**kwargs):
    print('beam_sim_main')
    os.system('jupyter notebook beam_simulation/beam_sim_gui.ipynb > /dev/null &')
