
import numpy as np
from progressbar import progressbar


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
