
import os
import sys
import pickle
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from mapa_holo import mapa_holo
from beam_design.beam_design_OLD import beam_design

# Paths
holos_path = Path(os.getcwd()).parent / 'Holograms'


def holo_generator(beam_type, filename=None, **kwargs):
    size = (768 // 2, 1024 // 2)
    if beam_type == 0:
        beam_name = 'SwitchOff'
        E_x = np.zeros(size)
        Ph_x = E_x
    else:
        E_x, E_y, Ph_x, Ph_y, beam_name = beam_design(size, beam_type, **kwargs)

    holo1 = mapa_holo(E_x, Ph_x, **kwargs)

    basename = beam_name if filename is None else filename

    imageio.imwrite(holos_path / (basename+'.png'), holo1)


if __name__ == '__main__':

    Y1 = -56
    Y2 = 38
    X1 = 1158
    X2 = 1254

    delta_Y = abs(Y1 - Y2)
    center_Y = (Y1 + Y2) // 2

    delta_X = abs(X1 - X2)
    center_X = (X1 + X2) // 2

    rho_max = (delta_X + delta_Y) // 4

    print('beam radius:', [delta_X, delta_Y, rho_max])
    print('beam center:', [center_X, center_Y])

    for modulation in ('complex', 'real', 'amplitude'):
        # holo_generator(0)
        # holo_generator(1)
        for sigma in [25]:  # [1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 150, 200]:
            holo_generator(87, NA=0.75, rho_max=rho_max, verbose=0, sigma=sigma,
                           ModulationType=modulation)
            # holo_generator(87, NA=0.75, rho_max=rho_max, verbose=0, sigma=sigma,
            #                ModulationType=modulation, topo=1)
            # holo_generator(87, NA=0.75, rho_max=rho_max, verbose=0, sigma=sigma,
            #                ModulationType=modulation, topo=-1)
