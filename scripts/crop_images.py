

from glob import glob
import imageio
import matplotlib.pyplot as plt
from os import path

SUFFIX = 0
PREFIX = 1
SAME = 2

description = "Crops images around Cx,Cy center and with a size of Sx and Sy " \
              "for a rectangular cropping. If just one size is passed then Sy=Sx."

def usage(msg=''):
    print("\n > " + (msg or description) + "\n")
    print("Use a '--pattern BASH_PATTERN' (required) to make a list of images to "
          "find it center. Finally, they will be cropped around that center.\n"
          "If '--save _SUFFIX/PREFIX_' is not passed, they will be OVERWRITTEN.\n"
          "If '--ref Cx,Cy,S' is passed, the irradiance of the cropping is normalized "
          "to the irradiance of the reference zone.\n"
          "If '--ref manual' then user must choose in the first image.\n")
    exit(int(bool(msg)))


def main(**kwargs):
    if kwargs.get('man') or not kwargs.get('pattern', ''):
        usage()

    name_opt = kwargs.get('save')
    suffix = ''
    prefix = ''
    if name_opt is not None:
        if name_opt[0] == '_':
            suffix = name_opt
        elif name_opt[-1] == '_':
            prefix = name_opt
        else:
            usage("The '--save' argument must start (suffix) or end (prefix) with "
                  "an underscore.")

    ref_opt = kwargs.get('ref', None)
    ref_x, ref_y, ref_s = None, None, None
    if ref_opt is not None and ref_opt != 'manual':
        ref = [int(r) for r in ref_opt.split(',')]
        if len(ref_opt) != 3:
            usage("The '--ref' argument must have three integers: Cx,Cy,S "
                  "-or- 'manual'.")
        else:
            ref_x, ref_y, ref_s = ref

    files = glob(kwargs.get('pattern'))

    if not input(f"{len(files)} files found to crop. Proceed? (y/N) ").lower().startswith('y'):
        exit()

    size = None
    for image_fn in files:

        tree = path.split(image_fn)
        fn_ext = path.splitext(tree[-1])
        cropped_fn = path.join(*tree[:-1], prefix + fn_ext[0] + suffix + fn_ext[1])

        im = imageio.imread(image_fn)  # .astype('float64')
        print("Find the center, close the window and type the center.")
        if size is None:
            print("Also check the desired windows size.")
        if ref_x is None:
            print("In addition, check the reference zone.")
        plt.imshow(im)
        plt.show()
        x, y = input("The center is at (x,y): ").split(',')
        if size is None:
            size = input("Desired windows size: ")
        if ref_opt is not None and ref_x is None:
            ref_x, ref_y, ref_s = input("The reference is at (x,y,size): ").split(',')

        if ref_opt:
            cropped = crop(im, int(x), int(y), int(size),
                           ref_x=int(ref_x), ref_y=int(ref_y), ref_s=int(ref_s))
        else:
            cropped = crop(im, int(x), int(y), int(size))

        print(f"Saving {path.basename(cropped_fn)}: {cropped.shape} "
              f"[{cropped.min():.2f}, {cropped.max():.2f}]")

        imageio.imwrite(cropped_fn, cropped.astype('uint16'))


def crop(im, cx, cy, sx, sy=None, **kwargs):
    """ Crop an image with center in (cx, cy) and size (sx, sy).
        If sy is not passed, a squared crop is done, i.e. sy=sy
    """
    if type(im) == str:
        im = imageio.imread(im)
    ref_x, ref_y, ref_s = kwargs.get('ref_x'), kwargs.get('ref_y'), kwargs.get('ref_s')
    ref_mean = 1
    if ref_x is not None:
        ref = im[ref_y - ref_s // 2:ref_y + ref_s // 2,
                 ref_x - ref_s // 2:ref_x + ref_s // 2]
        ref_value = ref.mean() / 500
    sy = sx if sy is None else sy
    return im[cy - sx // 2:cy + sx // 2, cx - sy // 2:cx + sy // 2] / ref_value