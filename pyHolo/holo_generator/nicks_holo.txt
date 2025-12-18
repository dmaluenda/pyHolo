
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio

from pyHolo.holo_generator.mapa_holo import mapa_holo

verbose = 1
initial_GL = 0
final_GL = 194
m_linphase = 0.25
linphase = "post"  # "pre" or "post"

slm_size = (1024, 1280)

x, y = np.meshgrid(np.linspace(-slm_size[1]//2, slm_size[1]//2, slm_size[1]),
                   np.linspace(slm_size[0]//2, -slm_size[0]//2, slm_size[0]))

field = np.cos(np.atan2(y[::2, ::2],x[::2, ::2])) + 0j

if linphase == "pre":
    field *= np.exp(1j * m_linphase * x[::2, ::2])


# holo = mapa_holo(np.abs(field), np.angle(field),
#                  ModulationType=f'HM{initial_GL}-{final_GL}',
#                  verbose=1, algorithm=0)

intrinsic_ph = np.linspace(0, 2*np.pi, final_GL-initial_GL+1)
dynamic_range = np.linspace(initial_GL, final_GL + 1, final_GL - initial_GL + 1,
                            dtype='uint8')  # 0-255 for 8-bit

# Generar totes les combinacions únicament on idx2 >= idx1
idx1, idx2 = np.triu_indices(len(intrinsic_ph))
ph1 = intrinsic_ph[idx1]
ph2 = intrinsic_ph[idx2]

print(ph1.shape, ph1.dtype)
print(idx1.shape, idx1.dtype)

# Aplicar el procediment d'Arrizon de manera vectoritzada
C_SLM = (np.exp(1j * ph1) + np.exp(1j * ph2)) / 2
Mapa1 = dynamic_range[idx1]
Mapa2 = dynamic_range[idx2]

if verbose > 1:
    plt.figure()
    plt.polar(ph_SLM, T_SLM, 'x')
    plt.title('SLM%d' % slm)
    plt.show()

N = field.shape
idx = np.zeros(field.shape, dtype='int')

for i in range(N[0]):
    for j in range(N[1]):
        idx[i, j] = np.argmin(abs(field[i, j] - C_SLM))


holo = np.zeros(np.dot(field.shape, 2), dtype='uint8')
holo[0::2, 0::2] = Mapa1[idx]
holo[1::2, 1::2] = Mapa1[idx]
holo[0::2, 1::2] = Mapa2[idx]
holo[1::2, 0::2] = Mapa2[idx]

if linphase == "post":
    holo_ph = np.exp(1j*holo/final_GL*2*np.pi)

    lin_phase = np.exp(1j * m_linphase * x)

    holo_ph *= lin_phase

    holo = np.astype(np.mod(np.angle(holo_ph), 2*np.pi)/2/np.pi*final_GL, "uint8")

plt.imshow(holo)
plt.colorbar()
plt.show()


holos_path = Path(os.getcwd()).parent.parent / 'pyHolo_userFolder' / 'Holograms'
basename = f"Hammamatsu_{initial_GL}-{final_GL}_m{m_linphase}{linphase}Arizon_2"

print(f"Saving {holos_path / (basename + '.npy')}")
print(f"Size of the hologram: {holo.shape}")
np.save(holos_path / (basename + ".npy"), holo)