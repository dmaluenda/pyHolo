
import os
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import numpy as np

directory = Path(os.getcwd()).parent.parent / "Images" / "Deep"

max_values = {}
for file in directory.iterdir():
    params = str(file).split('_')
    for param in params:
        if param.startswith('s'):
            sigma = int(param.split('s')[1])
        if param.startswith('z'):
            z = int(param.split('z')[1])

    im = imageio.imread(file, format='png')
    mValue = im.max()

    if sigma in max_values.keys():
        max_values[sigma].update({z: mValue})
    else:
        max_values[sigma] = {z: mValue}

data = sorted(max_values.items())
dict = {k: v for k, v in data if k not in (20, 70)}

plt.figure()
for sig in dict:
    sdict = sorted(dict[sig].items())
    x_data = [x for x, y in sdict]
    y_data = [y for x, y in sdict]
    plt.plot(x_data, y_data, label=sig)
plt.legend()
plt.show()