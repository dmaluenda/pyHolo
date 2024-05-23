
import os
import sys
import pickle
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt


from progressbar import ProgressBar, AnimatedMarker

from pathlib import Path
from PIL import Image

try:
    from .mapa_holo import mapa_holo_plain
except ImportError:
    from mapa_holo import mapa_holo_plain

# Paths
support_path = Path(os.getcwd()) / "gui" / "SupportFiles"


N = (768, 1024)

# These parameters should match in the labHolo.vi program
Tstep = 2
PHstep = 15
PHf = 361
Cstep = 5


def create_support_dir(dir_name):
    if not (support_path / dir_name).is_dir():
        print(f"Creating {dir_name} folder")
        (support_path / dir_name).mkdir()


def create_gray_level_images():
    create_support_dir('GrayLevel')
    create_support_dir('SemiGrayLevels')
    create_support_dir('SemiGrayLevelsV')

    print("Generating the plain gray level images:")
    pbar = ProgressBar()
    for gl in pbar(range(0, 256)):
        plain = np.ones(N, dtype='uint8') * gl
        imageio.imwrite(support_path / 'GrayLevel' / f"{gl}level.png", plain)

        semi = plain.copy()
        semi[:N[0] // 2, :N[1]] = 0
        imageio.imwrite(support_path / 'SemiGrayLevels' / f"{gl}semilevel.png", semi)

        semiV = plain.copy()
        semiV[:N[0], :N[1] // 2] = 0
        imageio.imwrite(support_path / 'SemiGrayLevelsV' / f"{gl}semilevel.png", semiV)


def create_basic_holos():

    create_support_dir('Holos')
    create_support_dir('SemiHolos')
    create_support_dir('TestPatterns')

    for SLM_number in (1, ):
        print(f"SLM{SLM_number}", flush=True)
        for modulation in ['complex', 'amplitude', 'real']:
            print(f"Modulation: {modulation}", flush=True)
            # widgets = ['Working: ', AnimatedMarker(markers='◢◣◤◥')]
            pbar = ProgressBar()
            for Trans in pbar(range(0, 101, Tstep)):
                for Phase in range(0, PHf, PHstep):
                    # Full holos
                    SLM_holo = mapa_holo_plain(Trans/100, Phase, N, SLM_number, modulation)
                    imageio.imwrite(support_path / 'Holos' /
                                    f"SLM{SLM_number}_T{Trans}_ph{Phase}_{modulation}.png", SLM_holo)

                    if Trans == 0:  # Switcher off
                        imageio.imwrite(support_path / 'TestPatterns' /
                                        f"SwitchOff_{SLM_number}_{modulation}.png", SLM_holo)
                    elif Trans == 100 and Phase == 0 and modulation == 'amplitude':
                        imageio.imwrite(support_path / 'TestPatterns' /
                                        f"KnifeEdge_{SLM_number}_{modulation}.png", SLM_holo)

                    # Semi holos
                    SLM_sholo = mapa_holo_plain(Trans/100, Phase, N, SLM_number, modulation, semi=True)
                    imageio.imwrite(support_path / 'SemiHolos' /
                                    f"SLM{SLM_number}_T{Trans}_ph{Phase}_{modulation}.png", SLM_sholo)


if '__main__' == __name__:
    _args = sys.argv

    if '--only-holos' in _args:
        create_basic_holos()
    elif '--only-gl' in _args:
        create_gray_level_images()
    else:
        create_gray_level_images()
        create_basic_holos()