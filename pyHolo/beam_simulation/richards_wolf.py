
import numpy as np
from matplotlib import patches
from tqdm import tqdm as progressbar

import matplotlib.pyplot as plt

class RichardsWolf:

    def __init__(self, EP_size=(512, 512), NA=1., rho_max=None):

        self.EP_size = EP_size
        self.NA = NA

        x, y = np.meshgrid(range(-EP_size[1] // 2, EP_size[1] // 2, ),
                           range(EP_size[0] // 2, -EP_size[0] // 2, -1))

        self.phi = np.mod(np.arctan2(y, x), 2 * np.pi)  # [0 2pi]
        self.cos_phi = np.cos(self.phi)
        self.sin_phi = np.sin(self.phi)

        rho_max = min(EP_size) / 2 if rho_max is None else rho_max
        rho = np.sqrt(x**2 + y**2)  # in pixels

        self.theta_0 = np.arcsin(NA)  # maximum angle
        print(f"{self.theta_0/np.pi=}")
        f_pix = rho_max / np.tan(self.theta_0)  # equivalent focal length in pixels
        print(f"{f_pix=}")
        self.theta = np.arctan2(f_pix, rho)  # angular radial coordinate

        self.cos_theta = np.cos(self.theta)
        self.sin_theta = np.sin(self.theta)
        self.aperture = self.sin_theta <= NA

        # un-normalized coordinates
        self.x = x
        self.y = y
        self.rho = rho

        # normalized coordinates to the rho_max (rho=1 => EP_limit)
        self.x_norm = x/rho_max
        self.y_norm = y/rho_max
        self.rho_norm = rho/rho_max

        # Charo's notation
        self.alpha = self.cos_theta
        self.alpha_0 = np.cos(self.theta_0)
        self.alpha_bar = (self.alpha_0 + 1) * 0.5

        self.rho_max = rho_max

    def imshow(self, ax, attr, cmap, vmin, vmax, title=None):
        title = title if title else attr
        obj = self.get(attr)
        ax.imshow(obj, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title+f'  [{obj.min():.1f}, {obj.max():.1f}]', fontsize=20)
        limit_EP = patches.Circle((self.EP_size[0]/2, self.EP_size[1]/2),
                                  self.rho_max, linestyle='--',
                                  edgecolor='red', facecolor='none')
        ax.add_patch(limit_EP)
        ax.axis('off')

    def imshow_EP(self):
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))

        to_show = [{'attr': 'x_norm', 'cmap': 'gray', 'vmin': -1, 'vmax': 1, 'title': r'$\frac{x}{\rho_{max}}$'},
                   {'attr': 'y_norm', 'cmap': 'gray', 'vmin': -1, 'vmax': 1, 'title': r'$\frac{y}{\rho_{max}}$'},
                   {'attr': 'rho', 'cmap': 'gray', 'vmin': 0, 'vmax': self.rho_max, 'title': r'$\rho$'},
                   {'attr': 'rho_norm', 'cmap': 'gray', 'vmin': 0, 'vmax': 1, 'title': r'$\rho_{norm}$'},
                   {'attr': 'phi', 'cmap': 'hsv', 'vmin': 0, 'vmax': np.pi*2, 'title': r'$\phi$'},
                   {'attr': 'theta', 'cmap': 'gray', 'vmin': 0, 'vmax': np.pi/2, 'title': r'$\theta$'},
                   {'attr': 'cos_theta', 'cmap': 'gray', 'vmin': 0, 'vmax': 1, 'title': r'$\cos(\theta)$'},
                   {'attr': 'sin_theta', 'cmap': 'gray', 'vmin': 0, 'vmax': 1, 'title': r'$\sin(\theta)$'},
                ]
        for ax, item in zip(axs.flatten(), to_show):
            self.imshow(ax, item['attr'], item['cmap'], item['vmin'], item['vmax'], item['title'])
        plt.tight_layout()
        plt.show()

        # ax = ax.T
        # ax[0, 0].imshow(self.x_norm, vmin=-1, vmax=1, cmap='gray')
        # ax[0, 0].set_title(f'x  [{self.x.min():.1f}, {self.x.max():.1f}]', fontsize=20)
        # ax[0, 1].imshow(self.y_norm, vmin=-1, vmax=1, cmap='gray')
        # ax[0, 1].set_title(f'y  [{self.y.min():.1f}, {self.y.max():.1f}]', fontsize=20)
        # ax[1, 0].imshow(self.rho_norn, cmap='gray', vmin=0, vmax=1)
        # ax[1, 0].set_title(r'$\frac{\rho}{\rho_{0}}$'+f"  [{self.rho_NA.min():.2f}, {self.rho_NA.max():.2f}]", fontsize=20)
        # ax[1, 1].imshow(self.phi, cmap='hsv', vmin=0, vmax=np.pi*2)
        # ax[1, 1].set_title(r'$\phi$'+f"  [{self.phi.min():.2f}, {self.phi.max():.2f}]", fontsize=20)
        # ax[2, 0].imshow(self.theta, cmap='gray', vmin=0, vmax=np.arcsin(self.NA))
        # ax[2, 0].set_title(r'$\theta$'+f"  [{self.theta.min()/np.pi:.2f}, {self.theta.max()/np.pi:.2f}]"+r" $\pi$", fontsize=20)
        # ax[2, 1].imshow(self.cos_theta, cmap='gray')
        # ax[2, 1].set_title(r'$\cos(\theta)$'+f"  [{self.cos_theta.min():.2f}, {self.cos_theta.max():.2f}]", fontsize=20)
        # ax[3, 0].imshow(self.sin_theta, cmap='gray', vmin=0, vmax=self.NA)
        # ax[3, 0].set_title(r'$\sin(\theta)$'+f"  [{self.sin_theta.min():.2f}, {self.sin_theta.max():.2f}]", fontsize=20)
        # ax[3, 1].imshow(self.aperture, cmap='gray', vmin=0, vmax=1)
        # ax[3, 1].set_title(r'$\sin(\theta) \leq NA$'+f"  [{self.aperture.min():.0f}, {self.aperture.max():.0f}]", fontsize=20)


    def get(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise Exception(f"Attribute '{attr}' not found!")

    def __str__(self):

        return (f"Richards-Wolf integrator of {self.EP_size} points,"
                f"NA={self.NA}.")


class Beam():

    def __init__(self, rw_integrator=None, EP_size=None, FZ_size=None):

        self.rw = rw_integrator if rw_integrator else RichardsWolf(EP_size)

    def set_EP_field(self, Ex, Ey):
        self.Ex = Ex
        self.Ey = Ey

    def set_EP_field_from_stokes(self, S0, S1, S2, S3):
        self.Ex = np.sqrt(S0 + S1)
        self.Ey = np.sqrt(S0 - S1)
        rel_phase = np.arctan2(S3, S2)
        self.Ey *= np.exp(1j * rel_phase)

    def get_EP_field(self, base='Cartesian'):

        if base == 'Cartesian':
            return self.Ex, self.Ey
        elif base == 'AzimuRadial':
            return self.rw.getAzimuRadialComps(self.Ex, self.Ey)
        else:
            print('Base is not supported! Choose a valid base '
                  '("Cartesian" or "AzimuRadial")')
            exit(1)

    def get_EV_field(self, base='Cartesian'):
        if self.rw is None:
            raise Exception('Richards-Wolf integrator is not set!')

        return self.rw.getVectorSpectrum(self.Ex, self.Ey, base)

    def get_FZ_field(self, z, base='Cartesian', force=False):
        if self.rw is None:
            raise Exception('Richards-Wolf integrator is not set!')

        if force or not hasattr(self, 'Ex_FZ'):
            self.Ex_FZ, self.Ey_FZ, self.Ez_FZ = self.rw.getFocusedField(self.Ex, self.Ey, z, base)
            return self.Ex_FZ, self.Ey_FZ, self.Ez_FZ
        else:
            return self.Ex_FZ, self.Ey_FZ, self.Ez_FZ
















class RichardsWolfIntegrator:

    def __init__(self, N, L, NA):

        x, y = np.meshgrid(np.linspace(-N/2,  N/2, N) / L,
                           np.linspace( N/2, -N/2, N) / L)
        self.x = x
        self.y = y
        self.N = N
        self.NA = NA

    def get_coords(self):
        rho2 = self.x * self.x + self.y * self.y
        rho = np.sqrt(rho2)
        mask = (rho <= self.NA) * 1

        theta = np.arcsin(rho * mask)
        phi = np.arctan2(self.y, self.x)

        return theta, mask, phi

    def getAzimuRadialComps(self, E0x, E0y):

        x_exit, y_exit = np.meshgrid(np.linspace(-1, 1, self.N),
                                     np.linspace(1, -1, self.N))

        phi = np.arctan2(y_exit, x_exit)
        sinfi = np.sin(phi)
        cosfi = np.cos(phi)

        # e1 vector (Azimuthal)
        e1x = - sinfi
        e1y = cosfi

        # ep vector (Radial on the Paraxial beam)
        epx = cosfi
        epy = sinfi

        # azimuthal and radial components
        f1 = E0x * e1x + E0y * e1y  # f1 = E0 · e1
        f2 = E0x * epx + E0y * epy  # f2 = E0 · e2

        return f1, f2


    def getFocusedField(self, Einc1, Einc2, z_array, base):

        z_array = np.array([z_array]) if isinstance(z_array, int) else z_array

        N = self.N
        if (N < 2 * self.NA ** 2 * np.abs(z_array).max() /
                         np.sqrt(1 - self.NA ** 2)):
            print('N is in undersampling conditions!')


        # Coordinates on the Reference Sphere
        theta, phi, mask = self.get_coords()

        # Spherical coordinates on the Reference Sphere
        sinfi = np.sin(phi) * mask
        cosfi = np.cos(phi) * mask
        sinte = np.sin(theta) * mask
        coste = np.cos(theta) * mask



        # Vectors and parameters of Richards - Wolf integral
        # appodization of isoplanatic an objective microscope
        P = np.sqrt(np.abs(coste))

        # e1 vector (Azimuthal)
        e1x = - sinfi * mask
        e1y = cosfi * mask
        e1z = 0

        # e2 vector (Radial on the Reference sphere)
        e2x = coste * cosfi * mask
        e2y = coste * sinfi * mask
        e2z = - sinte * mask

        # ep vector (Radial on the Paraxial beam)
        epx = cosfi * mask
        epy = sinfi * mask
        epz = 0

        # Angular spectrum of plane waves

        if base == 'Cartesian':
            f1 = Einc1 * e1x + Einc2 * e1y
            f2 = Einc1 * epx + Einc2 * epy
        elif base == 'AzimuRadial':
            f1 = Einc1
            f2 = Einc2
        else:
            print('Base is not supported! Choose a valid base '
                  '("Cartesian" or "AzimuRadial")')
            exit(1)

        EFx = np.zeros((N, N, len(z_array)), dtype=np.complex128)
        EFy = np.zeros((N, N, len(z_array)), dtype=np.complex128)
        EFz = np.zeros((N, N, len(z_array)), dtype=np.complex128)

        # wb = waitbar(0, 'Richards-Wolf integration via FFT. Z-loop:');

        eps = np.finfo(np.complex128).eps

        for iz, z_curr in enumerate(progressbar(z_array)):

            # Angular spectrum of plane waves
            Vx = P * (f1 * e1x + f2 * e2x) * np.exp(1j * 2 * np.pi * coste * z_curr)
            Vy = P * (f1 * e1y + f2 * e2y) * np.exp(1j * 2 * np.pi * coste * z_curr)
            Vz = P * (f1 * e1z + f2 * e2z) * np.exp(1j * 2 * np.pi * coste * z_curr)

            EFx[:, :, iz] = np.fft.fftshift(np.fft.fft2(
                               np.fft.ifftshift(Vx / (coste + eps) * mask)))
            EFy[:, :, iz] = np.fft.fftshift(np.fft.fft2(
                               np.fft.ifftshift(Vy / (coste + eps) * mask)))
            EFz[:, :, iz] = np.fft.fftshift(np.fft.fft2(
                               np.fft.ifftshift(Vz / (coste + eps) * mask)))

        return EFx, EFy, EFz
