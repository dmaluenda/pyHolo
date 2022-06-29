
import matplotlib.pyplot as plt
import imageio
import numpy as np
import regex


def crop(im, cx, cy, sx, sy=None, **kwargs):
    """ Crop an image with center in (cx, cy) and size (sx, sy).
        If sy is not passed, a squared crop is done, i.e. sy=sy
    """
    if type(im) == str:
        im = imageio.imread(im)
    sy = sx if sy is None else sy
    return im[cy - sx // 2:cy + sx // 2, cx - sy // 2:cx + sy // 2]



def main(prefix_fn, **kwargs):
    analizers = ('0', '90', '45', '135', 'Dex', 'Lev')  # The order is important
    cropping = kwargs.get('crop', None)
    # print(prefix_fn)
    reg_exp_patt = (f"^(?P<prefix>[a-zA-Z0-9_\\\/.]+)"
                    # f"_p(?P<polarizer>[a-zA-Z0-9]+)"  # put this in a separated regex
                    f"_a(?P<analizer>0|90|45|135|Dex|Lev)"
                    f"(?P<suffix>[a-zA-Z0-9_]*)"
                    f"(?P<extension>.[a-zA-Z]+)$")

    groups = [m for m in regex.finditer(reg_exp_patt, prefix_fn)]

    if not groups:
        root_path = '.'
        prefix_fn = prefix_fn
    else:
        gr_dict = groups[0].groupdict()
        # print(gr_dict)
        # polarizer = gr_dict['polarizer']
        prefix_fn = gr_dict['prefix']  # + '_p' + polarizer
        final = gr_dict['suffix'] + gr_dict['extension']

    for suff in analizers:
        filename = prefix_fn + '_a' + suff + final
        # print(filename)
        im = imageio.imread(filename).astype('float64')
        # print(im)
        if cropping:
            im = crop(im, *cropping)
        if suff == '0':
            buff = im
        elif suff == '90':
            stokes0 = buff + im
            stokes1 = buff - im
        elif suff == '45':
            buff = im
        elif suff == '135':
            stokes2 = buff - im
        elif suff == 'Dex':
            buff = im
        elif suff == 'Lev':
            stokes3 = buff - im

    fig_fn = kwargs.get('save', False)
    if kwargs.get('verbose', False) or fig_fn:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        stokes_list = [stokes0/stokes0.max(), stokes1/stokes0.max(),
                       stokes2/stokes0.max(), stokes3/stokes0.max()]
        beam_name = prefix_fn.split('\\')[-1]
        beam_name = '_'.join(beam_name.split('_')[-2:])
        fig.suptitle(fr"Stokes images for '{beam_name}' beam")
        for idx, ax in enumerate(axes.flatten()):
            stoke = stokes_list[idx]
            im = ax.imshow(stoke, vmin=-1, vmax=1, cmap='seismic')
            ax.set(title=rf"$S_{idx} (min:{stoke.min():.2f} ; max:{stoke.max():.2f}$")

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        if fig_fn:
            plt.savefig(fig_fn)
        plt.show()


        return im

