
import os
import sys
import pickle
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from .mapa_holo import mapa_holo_plain

# Paths
support_path = Path(os.getcwd()).parent / 'SupportFiles'


N = (768, 1024)

# These parameters should match in the LabView MAINEST.vi hologram loader app
Tstep = 2
PHstep = 15
PHf = 361
Cstep = 5


aux = 1
for SLM_number in (1, ):
    for modulation in ['complex', 'amplitude', 'real']:
        for Trans in range(0, 101, Tstep):
            for Phase in range(0, PHf, PHstep):
                # Full holos
                SLM_holo = mapa_holo_plain(Trans/100, Phase, N, SLM_number, modulation)
                imageio.imwrite(support_path / 'Holos' /
                                f"SLM{SLM_number}_T{Trans}_ph{Phase}_{modulation}.png", SLM_holo)

                if Trans == 0:  # Switcher off
                    imageio.imwrite(support_path / 'TestPatterns' /
                                    f"SwitchOff_{SLM_number}_{modulation}.png", SLM_holo)

                # Semi holos
                SLM_sholo = mapa_holo_plain(Trans/100, Phase, N, SLM_number, modulation, semi=True)
                imageio.imwrite(support_path / 'SemiHolos' /
                                f"SLM{SLM_number}_T{Trans}_ph{Phase}_{modulation}.png", SLM_sholo)
