
import matplotlib.pyplot as plt
import imageio
import numpy as np
import regex
from os import path
from .crop_images import crop

description = "Computes the Stokes images. It is computed from 6 images like " \
              "PREFIX_aXX_SUFFIX.EXT where XX is 0, 45, 90, 135, Dex are Lev. " \
              "FILENAME must match this structure."


def usage(msg=''):
    print("\n > " + (msg or description) + "\n")
    print(f"Usage: pyholo utils --script get_stokes --filename <myBeam_pXX_aXX_suffix.ext> "
          f"[--save <save_fn>] [--crop <Cx Cy Sx [Sy=Sx]>] [-v]\n\n"
          f"\t--filename: The template filename to find the six polarimetric images.\n"
          f"\t--save: The filename of the figure to be saved. If it starts or "
          f"ends with an underscore, it is understood as a suffix.\n"
          f"\t--crop: The cropping parameters.\n"
          f"\t-v: Verbose mode.\n")
    exit(int(bool(msg)))


def main(**kwargs):
    if kwargs.get('man'):
        usage()

    analizers = ('90', '0', '135', '45', 'Dex', 'Lev')  # The order is important
    cropping = kwargs.get('crop', None)
    filename = kwargs.get('filename', '')
    # print(prefix_fn)
    reg_exp_patt = (f"^(?P<prefix>[a-zA-Z0-9_\\\/.]+)"
                    f"_p(?P<polarizer>[a-zA-Z0-9]+)"  # put this in a separated regex
                    f"_a(?P<analizer>0|90|45|135|Dex|Lev)"
                    f"(?P<suffix>[a-zA-Z0-9_]*)"
                    f"(?P<extension>.[a-zA-Z]+)$")

    groups = [m for m in regex.finditer(reg_exp_patt, filename)]

    save_fn = kwargs.get('save', '')

    if not groups:
        usage("The filename does not match the pattern: ")
    else:
        gr_dict = groups[0].groupdict()
        # print(gr_dict)
        prefix_fn = gr_dict['prefix']  # + '_p' + polarizer
        parent_fn = path.dirname(prefix_fn)
        beam_name = path.basename(prefix_fn)
        polarizer = '_p' + gr_dict['polarizer']
        suffix_fn = gr_dict['suffix']
        ext_fn = gr_dict['extension']
        final = suffix_fn + ext_fn
        if save_fn.startswith('_') or save_fn.endswith('_'):
            fig_fn = prefix_fn + polarizer + '_' + save_fn.strip('_') + gr_dict['suffix'] + '.png'
        else:
            fig_fn = save_fn

    for curr_analizer in analizers:
        filename = prefix_fn + polarizer + '_a' + curr_analizer + final
        # print(filename)
        im = imageio.imread(filename).astype('float64')/4
        # print(im.max(), im.min())

        if True:
            ref_fn = path.basename(filename).strip('cropped_')
            print(ref_fn)
            ref = crop(path.join(path.dirname(filename), ref_fn),
                       926, 685, 200).astype('float64') / 4
            ref = np.average(ref)
            im /= ref

        # print(im)
        if cropping:
            im = crop(im, *cropping)

        if curr_analizer == '90':
            buff = im
        elif curr_analizer == '0':
            stokes0 = buff + im
            stokes1 = buff - im
        elif curr_analizer == '135':
            buff = im
        elif curr_analizer == '45':
            stokes2 = buff - im
        elif curr_analizer == 'Dex':
            buff = im
        elif curr_analizer == 'Lev':
            stokes3 = buff - im

    if kwargs.get('verbose', False) or fig_fn:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        stokes_list = np.array([stokes0/stokes0.max(), stokes1/stokes0.max(),
                                stokes2/stokes0.max(), stokes3/stokes0.max()])
        # beam_name = prefix_fn.split('\\')[-1]
        # beam_name = '_'.join(beam_name.split('_')[-2:])
        fig.suptitle(fr"Stokes images for '{beam_name+polarizer+suffix_fn}' beam")
        for idx, ax in enumerate(axes.flatten()):
            stoke = stokes_list[idx]
            im = ax.imshow(stoke, vmin=-1, vmax=1, cmap='seismic')
            ax.set(title=rf"$S_{idx} (min:{stoke.min():.2f} ; max:{stoke.max():.2f}$")

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        if fig_fn:
            plt.savefig(fig_fn)
            np.save(path.splitext(fig_fn)[0]+'.npy', stokes_list)
        plt.show()


        return im
