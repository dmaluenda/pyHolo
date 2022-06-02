
import os, sys
import numpy as np
import matplotlib.pyplot as plt


class BeamDesigner:


    def __init__(self, slm_size, **kwargs):

        self.slm_size = slm_size

        self.x, self.y = np.meshgrid(range(-slm_size[0]//2,  slm_size[0]//2),
                                     range( slm_size[1]//2, -slm_size[1]//2, -1))

        self.NA = kwargs.pop('NA', 1.)
        self.rho_max = kwargs.pop('rho_max', min(self.slm_size) / 2)
        self.theta_0 = np.arcsin(self.NA)
        self.rho_0 = self.rho_max / np.tan(self.theta_0)
        self.win_size = (np.count_nonzero(self.aperture()[self.slm_size[0]//2, :]),
                         np.count_nonzero(self.aperture()[:, self.slm_size[1]//2]))
        self.alpha_0 = np.cos(self.theta_0)
        self.alpha_bar = (self.alpha_0 + 1) * 0.5

        self.extra_vars = kwargs

    def aperture(self):
        _rho = np.sqrt(self.x ** 2 + self.y ** 2)
        _theta = np.arctan2(_rho, self.rho_0)
        return (np.sin(_theta) <= self.NA)*1

    def phi(self):  # [0 2pi]
        return np.mod(np.arctan2(self.y, self.x), 2 * np.pi) * self.aperture()

    def cos_phi(self):
        return np.cos(self.phi()) * self.aperture()

    def sin_phi(self):
        return np.sin(self.phi()) * self.aperture()

    def rho(self):
        return np.sqrt(self.x ** 2 + self.y ** 2) * self.aperture()

    def theta(self):
        return np.arctan2(self.rho(), self.rho_0) * self.aperture()

    def cos_theta(self):
        return np.cos(self.theta()) * self.aperture()

    def sin_theta(self):
        return np.sin(self.theta()) * self.aperture()

    def alpha(self):
        return self.cos_theta() * self.aperture()

    def imshow(self):
        figures = ['x', 'y', 'phi', 'theta', 'cos_theta', 'sin_theta',
                   'rho', 'aperture', 'text']
        rows = 3
        cols = 3
        fig, axes = plt.subplots(rows, cols)
        for idx, label in enumerate(figures):
            ax = axes[idx // cols, idx % cols]
            if label == 'text':
                vars_in_text = ['slm_size', 'NA', 'rho_max', 'rho_0', 'theta_0',
                                'alpha_0', 'alpha_bar', 'win_size']
                text_list = [f'{var} = {getattr(self, var)}' for var in vars_in_text]
                text_list += [f'extra: {k} = {v}' for k, v in self.extra_vars.items()]
                s = '\n'.join(text_list)
                ax.text(0.2, 0.5, s, ha='left', va='center')
                ax.axis('off')
                continue

            var = getattr(self, label) if label in ('x', 'y') else getattr(self, label)()
            cmap = 'gray'
            vmin = None if label in ('x', 'y') else 0
            vmax = None
            if label in ('cos_theta', 'sin_theta', 'aperture'):
                vmax = 1
            elif label in ('phi',):
                vmax = 2 * np.pi
            elif label in ('theta',):
                vmax = np.pi / 2

            im = ax.imshow(var, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(label)
            fig.colorbar(im, ax=ax)
        plt.show()

if __name__ == '__main__':
    print(f"WARNING: running {os.path.basename(sys.argv[0])} "
          f"as script should be just for testing.")

    beam = BeamDesigner((1024, 768), NA=0.75, rho_max=990//2)
    beam.imshow()
