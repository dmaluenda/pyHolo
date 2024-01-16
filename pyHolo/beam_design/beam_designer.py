



import numpy as np


class BeamDesigner:

    def __init__(self, slm_size, na=1., rho_max=None):
        x, y = np.meshgrid(range(-slm_size[1] // 2, slm_size[1] // 2, ),
                           range(slm_size[0] // 2, -slm_size[0] // 2, -1))

        self.x = x
        self.y = y

        self.phi = np.mod(np.arctan2(y, x), 2 * np.pi)  # [0 2pi]
        self.rho = np.sqrt(x ** 2 + y ** 2)

        self.cos_phi = np.cos(self.phi)
        self.sin_phi = np.sin(self.phi)

        self.rho_max = min(slm_size) / 2 if rho_max is None else rho_max
        self.theta_0 = np.arcsin(na) if na else np.pi / 2
        self.rho_0 = self.rho_max / np.tan(self.theta_0)
        self.theta = np.arctan2(self.rho, self.rho_0)
        self.cos_theta = np.cos(self.theta)
        self.sin_theta = np.sin(self.theta)
        self.aperture = self.sin_theta <= na
        self.win_size = (np.count_nonzero(self.aperture[slm_size[0] // 2, :]),
                         np.count_nonzero(self.aperture[:, slm_size[1] // 2]))

        # Charo's notation
        self.alpha = self.cos_theta
        self.alpha_0 = np.cos(self.theta_0)
        self.alpha_bar = (self.alpha_0 + 1) * 0.5




