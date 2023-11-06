
import os
import sys

from glob import glob

import imageio
import numpy as np
import matplotlib.pyplot as plt

fft = lambda image, shape: np.fft.fft2(image, shape)  # np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), shape))  #
ifft = lambda image, shape: np.fft.ifft2(image, shape)  # np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image), shape))  #

def check_files(filenames, show=True):

    images = []

    for filename in filenames:
        if filename.endswith('.png'):
            image = imageio.imread(filename)  # check if v2 is better
        elif filename.endswith('.npy'):
            image = np.load(filename)
        else:
            print(f"Unknown file type: {filename}")
            continue

        if show:
            plt.imshow(image)
            plt.colorbar()
            plt.title(f"{filename} [{image.max():.2f}]")
            plt.show()

        images.append(image)

    return images


def plot_max(images):
    maxs = [image.mean() for image in images]
    plt.plot(maxs)
    plt.show()


def find_center(original, piece):
    """Find the center of the piece in the original image using the cross-correlation.
    """
    ft_original = fft(original, shape=original.shape)
    ft_piece = fft(piece, shape=original.shape)
    corr = np.abs(ifft(ft_original.conj() * ft_piece, shape=original.shape))

    center = np.unravel_index(np.argmax(corr), corr.shape)
    plt.imshow(original)
    plt.title(f"Center: {center}")
    plt.show()
    return center


def crop_image(image, center, size):
    """Crop the image around the center with the given size.
    """
    x, y = center
    w, h = size, size
    return image[y-h//2:y+h//2, x-w//2:x+w//2]


def find_center_and_crop(originals, pieces, size, show=True):
    """ Find the center of the piece in the original image using the cross-correlation.
    """
    recropped = []
    for original, piece in zip(originals, pieces):
        center = find_center(original, piece)
        new_image = crop_image(original, center, size)

        if show:
            plt.imshow(new_image)
            plt.colorbar()
            plt.title(f"Center: {center}")
            plt.show()

        recropped.append(new_image)

    return recropped


if "__main__" == __name__:
    args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python check_images.py <path_pattern_to_images>")
        sys.exit(1)

    path_pattern = args[0]
    print(args)
    show = '--no-show' not in args

    crop = False
    for arg in args:
        if arg.startswith('--crop'):
            crop = True
            size = int(arg.split('=')[1])
            print(f"Re-cropping to {size}x{size}")

        if arg.startswith('--originals'):
            originals = arg.split('=')[1]
            print(f"From {originals}")

    print(path_pattern)
    filenames = [f for f in glob(path_pattern)]
    print(f"Found {len(filenames)} files")

    images = check_files(filenames, show=show)
    plot_max(images)

    if crop:
        originals = check_files([originals], show=False)
        recropped = find_center_and_crop(originals, images, size, show=show)



