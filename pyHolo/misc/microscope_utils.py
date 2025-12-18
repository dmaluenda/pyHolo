from ipywidgets import IntSlider, Checkbox, Layout, Dropdown, fixed, VBox, HBox, interactive
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from pathlib import Path
import numpy as np

from .report_utils import print_fig


class ValueWithUnits:
    """ Class to store the pixel size of the camera with its units.
    """
    wavelen = 0.514  # in [um]
    units_values = {'nm': 1e-3, 'um': 1, 'mm': 1e3, 'lam': wavelen}
    units_alt = {'nm': 'nm', r'$\mu m$': 'um', 'mm': 'mm', r'$\lambda$': 'lam'}
    unit_def = 'um'

    @classmethod
    def set_wavelen(cls, wavelen):
        print(f"WARNING: The wavelength is changed from {cls.wavelen} to {wavelen}.\n"
              f"         This will affect EVERY object of {cls.__name__}.")
        cls.wavelen = wavelen

    @classmethod
    def assert_units(cls, units):
        possibles = list(cls.units_values.keys()) + list(cls.units_alt.keys())
        units_str = "', '".join(possibles)
        assert units in possibles+[None], f"Units must be '{units_str}' or None, not {units}"
        if units in cls.units_alt.keys():
            units = cls.units_alt[units]
        return units

    @classmethod
    def set_wavelen(cls, unit_def):
        cls.assert_units(unit_def)
        cls.unit_def = unit_def
    def __init__(self, value=None, units=None):
        units = self.assert_units(units)

        self.value = value
        self.units = units

    def __str__(self):
        return f"{self.value:.3f} {self.units}/pixel"

    def get(self, units=None):
        if self.value is None:
            print("Pixel size is not defined yet.")
            return -9999.99
        if units is None:
            print(f"WARNING: units are not defined, so pixel size is returned in {self.units}")
            return self.value
        else:
            return self.value * self.units_values[self.units] / self.units_values[units]

    def set(self, value, units=None):
        units = self.assert_units(units)
        if units is None:
            units = unit_def
            print(f"WARNING: units are not defined, so pixel size is set in '{self.unit_def}'")
        self.value = value
        self.units = units

    def set_units(self, units):
        units = self.assert_units(units)
        if self.units is not None and self.value is not None:
            self.value = self.get(units)
        self.units = units


def get_pixel_size_kwargs(scale_path, pixel_size, cXv, cYv, cXh, cYh, **inkwargs):
    scale_path = Path("data") / "scale.png" if scale_path is None else scale_path
    y_max, x_max = plt.imread(scale_path)[:, :, 0].shape
    main_slider = (lambda v, d, m: IntSlider(min=0, max=m, step=1, value=v,
                                          description='Xv',
                                          layout=Layout(width='500px'),
                                          continuous_update=False))
    sec_sliders = (lambda v, d: IntSlider(min=50, max=2000, step=10, value=v,
                                          description=d,
                                          layout=Layout(width='300px'),
                                          continuous_update=False))

    doCalc = pixel_size is not None
    kwargs = {}
    dropdown = lambda o, v, d: Dropdown(options=o, value=v, description=d,
                                        layout=Layout(width='auto',
                                                      description_width='400px'))
    kwargs.update(x_vert=main_slider(cXv, 'X vertical', x_max),
                  y_vert=main_slider(cYv, 'Y vertical', y_max),
                  x_hori=main_slider(cXh, 'X horizontal', x_max),
                  y_hori=main_slider(cYh, 'Y horizontal', y_max),
                  mean_size=sec_sliders(400, 'short size'),
                  axis_size=sec_sliders(700, 'long size'),
                  usaf_group=dropdown(list(range(9)), 7, 'Group'),
                  usaf_element=dropdown(list(range(9)), 1, 'Element'),
                  scale_path=fixed(scale_path),
                  units=dropdown(['lam', 'um', 'nm'], 'um', 'Units',),
                  lamb=dropdown([.488, .514, .633], .514, 'lam (um)'),
                  doCalculate=Checkbox(doCalc, description='Calculate'),
                  verbose=fixed(inkwargs.get('verbose', 3)),
                  fig_num=fixed(inkwargs.get('fig_num', 0)))

    return kwargs


def interactive_pixel_size(*args, **inkwargs):

    kwargs = get_pixel_size_kwargs(*args, **inkwargs)

    print(f"Let's check the resolution of the system by exploring "
          f"{inkwargs.get('scale_path', 'scale.png')} file, "
          f"i.e. the effective pixel size (sampling rate).\n\n"
          f"Unckeck 'Calculate' to be more responsive when interacting with sliders."
          f"") if inkwargs.get('verbose', 3) else None


    w = interactive(get_pixel_size, **kwargs)


    if inkwargs.get('verbose', 3):
        ch = w.children
        left_box = VBox([ch[0], ch[2]])
        right_box = VBox([ch[1], ch[3]])
        sl_box = HBox([left_box, right_box])
        opt_box = HBox(ch[4:8])
        opt_box2 = HBox(ch[8:-1])
        layout = VBox([sl_box, opt_box, opt_box2, ch[-1]])
        display(layout)

    return w


def get_pixel_size(x_vert, y_vert, x_hori, y_hori, mean_size=400, axis_size=700,
                   usaf_group=7, usaf_element=1, scale_path=None,
                   doCalculate=True, units='lam', lamb=.514, verbose=0, fig_num=0):
    line_pair_resolution = int( 2**(usaf_group +(usaf_element - 1) /6))

    scale_path = Path("data") / "scale.png" if scale_path is None else scale_path

    img_scale = plt.imread(scale_path)[:, :, 0]


    if doCalculate:
        img_scale -= img_scale.min()
        img_scale /= img_scale.max()
        fft_ = np.fft.fft2(img_scale)
        edge = 2
        fft_[edge+1:-edge, edge+1:-edge] = 0
        fft_[edge+2:-edge-1 ,:] = 0
        fft_[:, edge+2:-edge -1] = 0
        filter = np.abs(np.fft.ifft2(fft_))
        img_scale = img_scale / filter
        img_scale[np.where(img_scale > 1)] = 1

    vert_zone = (x_vert - axis_size//2, x_vert + axis_size//2,
                 y_vert - mean_size//2, y_vert + mean_size//2)
    hori_zone = (x_hori - mean_size//2, x_hori + mean_size//2,
                 y_hori - axis_size//2, y_hori + axis_size//2)


    zone_v = img_scale[vert_zone[2]:vert_zone[3], vert_zone[0]:vert_zone[1]]
    vert_profile = zone_v.mean(axis=0)

    zone_h = img_scale[hori_zone[2]:hori_zone[3], hori_zone[0]:hori_zone[1]]
    hori_profile = zone_h.mean(axis=1)

    pair_size_pixels = None
    pixel_size = ValueWithUnits(None, 'um')
    if doCalculate:
        halves = [np.where(np.diff(np.sign(vert_profile - 0.5)))[0],
                  np.where(np.diff(np.sign(hori_profile - 0.5)))[0]]

        delta2_vert = halves[0][4] - halves[0][0] + 1 if doCalculate else 0
        delta2_hori = halves[1][4] - halves[1][0] + 1 if doCalculate else 0

        pair_size_pixels = np.mean([delta2_vert, delta2_hori]) // 2
        pixel_size_um = 1000 / pair_size_pixels / line_pair_resolution  # in [um]
        pixel_size = ValueWithUnits(pixel_size_um, 'um')
        pixel_size.set_units(units)

    if verbose > 1:
        pitch = 1 if doCalculate else 4
        fig, img_ax = plt.subplots(1, figsize=(10 ,17))
        img_ax.imshow(img_scale[::pitch, ::pitch], cmap='gray')
        rectV = patches.Rectangle((vert_zone[0]/pitch, vert_zone[2]/pitch),
                                  axis_size/pitch, mean_size/pitch, linewidth=2,
                                  label="Vertical profile zone",
                                  edgecolor='r', facecolor="none")
        img_ax.add_patch(rectV)
        rectH = patches.Rectangle((hori_zone[0]/pitch, hori_zone[2]/pitch),
                                  mean_size/pitch, axis_size/pitch, linewidth=2,
                                  label="Vertical profile zone",
                                  edgecolor='g', facecolor="none")
        img_ax.add_patch(rectH)
        fig.legend(loc=4, bbox_to_anchor=(0.6, 0.38), facecolor='black', labelcolor='white')
        img_ax.set_xticklabels(img_ax.get_xticks()*pitch)
        img_ax.set_yticklabels(img_ax.get_yticks()*pitch)

        fig, axs = plt.subplots(1, 2, figsize=(10 ,5))
        ax = axs.flatten()
        for idx, profile in enumerate([vert_profile, hori_profile]):
            color = 'g' if idx else 'r'
            ax[idx].plot(profile, color)  #, label="Vertical profile"

            if doCalculate:
                half = halves[idx]
                delta_pixel2 = delta2_vert if idx else delta2_hori
                ax[idx].plot(half, profile[half], color + '.')
                for pos in half:
                    ax[idx].text(pos, profile[pos], f" {'y' if idx else 'x'}={pos}", verticalalignment='center')
                ax[idx].plot([half[0], half[4]], [0, 0], 'g-', linewidth=4)
                ax[idx].plot([half[0], half[0]], [0, profile[half[0]]], 'g-')
                ax[idx].plot([half[4], half[4]], [0, profile[half[4]]], 'g-')
                ax[idx].text(half[0] + 20, 0.04, f"{delta_pixel2} px / 2 = {delta_pixel2 / 2:.0f} pixels",
                             weight='bold', bbox=dict(boxstyle="round4", fc="w"))
            ax[idx].set_xlim([0, len(profile)])
            ax[idx].set_xlabel(f"{'Y' if idx else 'X'} pixel pos.")
            ax[idx].set_ylabel("Gray level")
            number_ticks = 4
            yticks_ = [x / number_ticks for x in range(number_ticks + 1)]
            ax[idx].set_yticks(yticks_)
            ax[idx].set_yticklabels([f"{x*255:.0f}" for x in yticks_])
            
        ax[1].yaxis.tick_right()
        plt.show()
        fig_num = print_fig(f'1951USAF Resolution test  '
                            f'[Group {usaf_group}, Element {usaf_element}] : '
                            f'line pair resolution = {line_pair_resolution}/mm', fig_num)
    print(f"pixel_size = 1e3 / pair_size_pixels / line_pair_resolution = "
          f"1000 / {pair_size_pixels} / {line_pair_resolution} = "
          f"{pixel_size.get('um'):.3f} um/px{(' = '+str(pixel_size)) if units != 'um' else ''}") if verbose else None

    return pixel_size, fig_num