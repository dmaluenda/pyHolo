
import os
import sys
import pickle
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt

from time import time

from pathlib import Path
from PIL import Image

try:  # Using pyHolo
    from .mapa_holo import mapa_holo
except ImportError:  # Standalone version with cwd as pyHolo's root directory
    from mapa_holo import mapa_holo
    sys.path.append(os.getcwd())

from beam_design.beam_design_OLD import beam_design as beam_design_old
import beam_design


# Paths
holos_path = Path(os.getcwd()) / 'Holograms'


def get_beam(beam_name, **kwargs):
    print(beam_name)
    BeamClass = getattr(beam_design, beam_name, None)
    return BeamClass(**kwargs)


def holo_generator(beam_type, filename=None, **kwargs):
    size = (768 // 2, 1024 // 2)
    if beam_type == 0:
        beam_name = 'SwitchOff'
        E_x = np.zeros(size)
        Ph_x = E_x
    else:
        E_x, E_y, Ph_x, Ph_y, beam_name = beam_design_old(size, beam_type, **kwargs)

    # t0 = time()
    holo1 = mapa_holo(E_x, Ph_x, **kwargs)
    # t1 = time()
    basename = beam_name if filename is None else filename

    imageio.imwrite(holos_path / (basename+".png"), holo1)
    # t2 = time()
    # print(f"mapa_holo: {t1-t0:.2f}s")
    # print(f"imwrite:   {t2-t1:.2f}s")

def holo_gen_main(**kwargs):
    rho_max = None  # Something related with EP_edges arg (see holo_generator.py)
    rho_max = kwargs.get('rho_max', None) if kwargs.get('rho_max', None) else rho_max

    beam = get_beam(kwargs.get('beam_type'), **kwargs)
    E_x, Ph_x, E_y, Ph_y = beam.get_field()
    beam_name = beam.beam_name

    if kwargs.get('verbose'):
        beam.geometry.imshow()
        beam.imshow()


if __name__ == '__main__':

    Y1 = -65
    Y2 = 119
    X1 = 1690
    X2 = 1872

    delta_Y = abs(Y1 - Y2)     # Vertical diameter
    center_Y = (Y1 + Y2) // 2  # Vertical center

    delta_X = abs(X1 - X2)     # Horizontal diameter
    center_X = (X1 + X2) // 2  # Horizontal center

    rho_max = (delta_X + delta_Y) // 4  # Average radius

    print('beam radius:', [delta_X, delta_Y, rho_max])
    print('beam center:', [center_X, center_Y])

    for modulation in ['complex']:  # ('complex', 'real', 'amplitude'):  #  ['amplitude']:  #
        print('Modulation:', modulation)
        # holo_generator(0)
        # holo_generator(1)
        t0 = time()
        for sigma in [-1]:  # range(100):  #  [1, 2, 3, 5, 10, 20, 25, 30, 50, 70, 100, 150, 200]:
            # print(f"Going for sigma={sigma}")
            # holo_generator(87, NA=0.75, rho_max=rho_max, verbose=2, sigma=sigma,
            #                ModulationType=modulation, filename='sigma2_for_paper')
            holo_generator(87, NA=0.75, rho_max=rho_max, verbose=0, sigma=sigma,
                           ModulationType=modulation, topo=1, algorithm=3)
            holo_generator(87, NA=0.75, rho_max=rho_max, verbose=0, sigma=sigma,
                           ModulationType=modulation, topo=-1, algorithm=3)
            # holo_generator(87, NA=0.75, rho_max=rho_max, verbose=1, sigma=sigma,
            #                ModulationType=modulation, topo=0)

        t1 = time()
        print(t1-t0)