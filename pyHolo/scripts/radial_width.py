
from beam_simulation.rhichards_wolf import RichardsWolfIntegrator
import numpy as np
import matplotlib.pyplot as plt

description = ""

def usage(msg=''):
    print("\n > " + (msg or description) + "\n")
    print("")
    exit(int(bool(msg)))


def main(**kwargs):

    rw_integrator = RichardsWolfIntegrator(N=1025, L=18, NA=0.5)

    theta, mask, phi = rw_integrator.get_coords()

    Ex_rad = np.cos(phi) * mask
    Ey_rad = np.sin(phi) * mask

    Er_rad = mask
    Ea_rad = np.zeros_like(mask)

    Ex_f, Ey_f, Ez_f = rw_integrator.getFocusedField(Ex_rad, Ey_rad, 0, base='Cartesian')
    print(np.isnan(Ex_f).all(),
          np.isnan(Ey_f).all(),
          np.isnan(Ez_f).all())
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(np.abs(Ex_f))
    ax[0, 0].set_title('Ex_f')
    ax[0, 1].imshow(np.abs(Ey_f))
    ax[0, 1].set_title('Ey_f')
    ax[1, 0].imshow(np.abs(Ez_f))
    ax[1, 0].set_title('Ez_f')
    ax[1, 1].imshow(np.sqrt(np.abs(Ex_f)**2 + np.abs(Ey_f)**2 + np.abs(Ez_f)**2))
    ax[1, 1].set_title('E_f')
    plt.show()




