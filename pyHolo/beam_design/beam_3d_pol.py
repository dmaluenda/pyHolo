



import numpy as np
from beam_designer import BeamDesigner


def beam_3d_pol(**kwargs):

    beamNAME = kwargs.get('beam_type', 'new')  # usually an integer
    modulation = kwargs.get('ModulationType', None)
    suffix_name = '_' + modulation if modulation is not None else ''

    bd = BeamDesigner((1024, 768), na=0.75, )

    sigma = kwargs.get('sigma', 5)
    topo = kwargs.get('topo', 0)
    rhoMult = kwargs.get('avoidCenter', False)

    beamNAME += '_s' + str(sigma)
    if topo:
        beamNAME += '_m' + str(topo)
    if rhoMult and not topo:
        beamNAME += '_rho'

    diff_alpha_2 = (bd.alpha - bd.alpha_bar) ** 2
    denominator = (1 - bd.alpha_0) ** 2

    alpha_fact = diff_alpha_2 / denominator

    g_alpha = (1 / (np.pi * np.sqrt(bd.alpha) * (1 + bd.alpha)) *
               np.exp(-(sigma / 2) * alpha_fact))

    E_x = g_alpha
    Ph_x = topo * bd.phi

    if topo > 0 or rhoMult:
        E_x *= bd.rho / bd.rho_max

    print('All elements are positive?', (E_x * np.exp(1j * Ph_x) >= 0).all())

