
import os, sys, io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


class EntrancePupilGeometry:

    def __init__(self, slm_size=(1024, 768), **kwargs):
        print(kwargs)
        self.slm_size = slm_size

        self.x, self.y = np.meshgrid(range(-slm_size[0]//2,  slm_size[0]//2),
                                     range( slm_size[1]//2, -slm_size[1]//2, -1))

        self.NA = kwargs.pop('NA', 1.)
        self.rho_max = kwargs.pop('rho_max', min(self.slm_size) / 2)
        print(self.rho_max)
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

    def crop(self, img):
        img[np.where(self.aperture()==0)] = 0
        return img

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


class BeamBase:

    def __init__(self, **kwargs):
        """ Base class for beam design.
            :param kwargs: keyword arguments
        """
        self.beam_name = kwargs.pop('beam_name', type(self).__name__)
        self.geometry = EntrancePupilGeometry(**kwargs)
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        pass

    def get_x_comp(self):
        return np.zeros_like(self.geometry.x), np.zeros_like(self.geometry.x)

    def get_y_comp(self):
        return np.zeros_like(self.geometry.x), np.zeros_like(self.geometry.x)

    def get_field(self):
        e_x, phi_x = self.get_x_comp()
        e_y, phi_y = self.get_y_comp()
        return (self.geometry.crop(e_x), self.geometry.crop(e_y),
                self.geometry.crop(phi_x), self.geometry.crop(phi_y))

    def get_beam_plot(self):
        e_x, e_y, phi_x, phi_y = self.get_field()

        fig = plt.figure(figsize=(20,10))
        axs = ImageGrid(fig, 111, nrows_ncols=(1, 2), cbar_mode="each")

        axs[0].imshow(np.abs(e_x))
        axs[0].set_title(r'Design: $|Ex|$')
        axs[1].imshow(np.angle(e_x))
        axs[1].set_title(r'Design: $\phi_x$')

        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()

        return data.reshape((int(h), int(w), -1))

    def imshow(self):
        
        e_x, e_y, ph_x, ph_y = self.get_field()
        
        I = (abs(e_x)) ** 2 + (abs(e_y)) ** 2
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].imshow(I, vmin=0, vmax=1)
        axs[0, 0].set_title('Design: I')
        axs[1, 0].imshow(e_x, vmin=0, vmax=1)
        axs[1, 0].set_title('Design: Ex')
        axs[1, 1].imshow(e_y, vmin=0, vmax=1)
        axs[1, 1].set_title('Design: Ey')
        axs[2, 0].imshow(ph_x, vmin=0, vmax=2 * np.pi)
        axs[2, 0].set_title('Design: ph_x')
        axs[2, 1].imshow(ph_y, vmin=0, vmax=2 * np.pi)
        axs[2, 1].set_title('Design: ph_y')
        ph = np.mod(ph_y - ph_x, 2 * np.pi)
        axs[0, 1].imshow(ph, vmin=0, vmax=2 * np.pi)
        axs[0, 1].set_title('Design: Ph')
        plt.show()

def get_beam_hologram(beam_name, *args, **kwargs):
    beam_class = globals()[beam_name]

    beam = beam_class(*args, **kwargs)

    beam_plot = beam.get_beam_plot()



    return

if __name__ == '__main__':
    print(f"WARNING: running {os.path.basename(sys.argv[0])} "
          f"as script should be just for testing.")

    beam = EntrancePupilGeometry((1024, 768), NA=0.75, rho_max=990 // 2)
    beam.imshow()

