
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
# from scipy.fft import fft2, ifft2, fftshift, ifftshift


def print_fig(msg, fig_num):
    fig_num += 1
    print(f"Figure {fig_num}: {msg}")
    return fig_num
    qtpydeqw


def plot_polarimetric_images(irradiances, label='', fig_num=0, cmap="gray"):
    # irradiances = [retriever_obj.cropped[0][pol] for pol in range(6)]
    maximum = max([irr.max() for irr in irradiances])

    pol_keys=["$I_{0}$", "$I_{45}$", "$I_{90}$", "$I_{135}$", "$I_{Lev}$", "$I_{Dex}$"]

    fig = plt.figure(figsize=(20, 30))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2, 3),
                    axes_pad=0.6,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.2
                    )

    for idx, ax in enumerate(axs):
        irr = irradiances[idx] if idx < len(irradiances) else np.zeros_like(irradiances[0])
        im = ax.imshow(irr/maximum, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(pol_keys[idx])

    cbar = plt.colorbar(im, cax=axs.cbar_axes[0], ticks=[0, 1])
    [t.set_fontsize(20) for t in cbar.ax.get_yticklabels()]
    plt.show()
    title = f"{label} beam: Polarimetric images."
    return print_fig(title, fig_num)


def plot_trans_field(Ex, Ey, label='', fig_num=0, cmap_abs='jet', cmap_ph='hsv'):

    Ax = np.abs(Ex)
    Ay = np.abs(Ey)

    A_T = np.sqrt(Ax**2 + Ay**2)

    fig = plt.figure(figsize=(20, 30))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2, 3),
                    axes_pad=0.6,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.2
                    )

    ax[0].imshow(Ax, cmap=cmap_abs)
    ax[0].set_title(r"$|Ex|$")
    ax[1].imshow(Ay, cmap=cmap_abs)
    ax[1].set_title(r"$|Ey|$")
    ax[1].imshow(A_T, cmap=cmap_abs)
    ax[1].set_title(r"$\sqrt{I_T}$")
    ax[3].imshow(np.angle(Ex), cmap=cmap_ph)
    ax[3].set_title(r"$\phi_x$")
    ax[4].imshow(np.angle(Ey), cmap=cmap_ph)
    ax[4].set_title(r"$\phi_y$")
    ax[5].imshow(np.angle(Ey/Ex), cmap=cmap_ph)
    ax[5].set_title(r"$\phi=\phi_y-\phi_x$")
    plt.show()
    title = f"{label} beam: Tranversal components of the field."
    return print_fig(title, fig_num)


def plot_long_field(Ez, label='', fig_num=0, cmap_abs='jet', cmap_ph='hsv'):
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(10,20))
    ax[0].imshow(np.abs(Ez), cmap=cmap_abs)
    ax[0].set_title(r"$|Ez|$")
    ax[1].imshow(np.angle(Ez), cmap=cmap_ph)
    ax[1].set_title(r"$\phi_z$")
    plt.show()
    title = f"({label} Pol.) The raw data of the retrieved longitudinal component."
    return print_fig(title, fig_num)


def plot_fields(total_field, cmp_phase_shift=None, fig_num=0, trim=None, label="",
                verbose=1, pixel_size=1, lamb=.514,
                ticks_step=1, units='lam'):
    """

    :param total_field: [Ex, Ey, Ez] complex fields
    :param cmp_phase_shift: polarization phase shift: np.arctan2(S[3], S[2]) for comparission porpouses
    :param fig_num:
    :param trim:
    :param label:
    :param verbose:
    :param pixel_size: in lambdas
    :param lamb: in um
    :param ticks_step: in units
    :param units: 'lam' (default), 'um', 'px'  (assuming the pixel size is in lambdas/pixel)
    :return: The updated fig_num
    """
    ntrim = -trim if trim else None  # negative trim (cicle/reverse slicing)

    normalize = lambda a: a / a.max()  # (a-a.min())/(a.max()-a.min())
    phase_diff = lambda ph1, ph2: np.angle(np.exp(1j*ph1)/np.exp(1j*ph2))  # ph1-ph2

    # Field components (assumed complex)
    Ex = total_field[0][trim:ntrim, trim:ntrim]
    Ey = total_field[1][trim:ntrim, trim:ntrim]
    Ez = total_field[2][trim:ntrim, trim:ntrim]

    # Amplitudes
    Ax = np.abs(Ex)
    Ay = np.abs(Ey)
    Az = np.abs(Ez)
    A_Total = np.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2)
    A_Trans = np.sqrt(Ax ** 2 + Ay ** 2)
    maxAT = np.max(A_Total)

    # Phases   [-pi, pi]
    phx = np.angle(Ex[trim:ntrim, trim:ntrim])
    phy = np.angle(Ey[trim:ntrim, trim:ntrim])
    phz = np.angle(Ez[trim:ntrim, trim:ntrim])
    phT = phase_diff(phy, phx)  # phy - phx
    cmp_phase_shift = (cmp_phase_shift[trim:ntrim, trim:ntrim]
                       if cmp_phase_shift else np.angle(Ey))
    ph_error = phase_diff(phT-cmp_phase_shift)  # phT - exp_phase_shift


    # Plotting parameters (sizes, units, extensions, ticks...)
    lims = Ex[trim:ntrim, trim:ntrim].shape
    scale_units = pixel_size if units == 'lam' else pixel_size*lamb if units == 'um' else 1
    extension = lims[0] * scale_units  # window size in units
    half_side = extension / 2  # in units
    extent = [-half_side, half_side, -half_side, half_side]  # in units
    str_units = r'\lambda' if units == 'lam' else r'\mu m' if units == 'um' else 'pixels'

    ticks = [0]  # central tick
    for tt in range(int(extension / ticks_step / 2)):
        # adding ticks in both directions
        idx = tt + 1
        ticks.append(idx * ticks_step)
        ticks.insert(0, -idx * ticks_step)

    # Colormaps
    my_greens = np.zeros((256, 4))
    my_greens[:, 1] = np.linspace(0, 1, 256)
    my_greens[:, 3] = 1
    cmap_amps = ListedColormap(my_greens)
    cmap_phs = 'twilight_shifted'

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

    fig = plt.figure(figsize=(20, 40))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2, 5),
                    axes_pad=0.6,
                    cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="5%",
                    cbar_pad=0.2
                    )

    for idx, ax in enumerate(axs):
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$ (in $'+str_units+'$)', fontsize=20)
        ax.set_ylabel(r'$y$ (in $'+str_units+'$)', fontsize=20)

        if idx == 0:
            im = ax.imshow(normalize(Ax), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title('|E_x|', Ax/maxAT), fontsize=fs)
        elif idx == 1:
            im = ax.imshow(normalize(Ay), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title('|E_y|', Ay/maxAT), fontsize=fs)
        elif idx == 2:
            im = ax.imshow(normalize(A_Trans), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title(r'\sqrt{|E_x|^2+|E_y|^2}', A_Trans/maxAT), fontsize=fs)
        elif idx == 3:
            im = ax.imshow(normalize(Az), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title('|E_z|', Az/maxAT), fontsize=fs)
        elif idx == 4:
            im = ax.imshow(normalize(A_Total), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title(r'\sqrt{I_T}', A_Total/maxAT), fontsize=fs)
        elif idx == 5:
            im = ax.imshow(phx, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta_x$', fontsize=fs)
        elif idx == 6:
            im = ax.imshow(phy, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta_y$', fontsize=fs)
        elif idx == 7:
            im = ax.imshow(phT, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta=\delta_y-\delta_x$', fontsize=fs)
        elif idx == 8:
            im = ax.imshow(phz, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta_z$', fontsize=fs)
        elif idx == 9 and trans_stokes:
            im = ax.imshow(ph_error, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\epsilon_\delta=\delta-\delta_{exp}$', fontsize=fs)


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
    return print_fig(f"{label} beam: Total field in the focal plane.", fig_num)