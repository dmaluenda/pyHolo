


import numpy as np
import matplotlib.pyplot as plt

from .designer import EntrancePupilGeometry
from .designer import BeamBase


class BeamUncertainPrinciple(BeamBase):

    def set_defaults(self, **kwargs):
        self.extra_vars = kwargs
        self.sigma = kwargs.pop('sigma', 5)
        self.topo = kwargs.pop('topo', 0)
        self.rhoMult = kwargs.pop('avoidCenter', False)

        self.beam_name += '_s' + str(self.sigma)
        if self.topo:
            self.beam_name += '_m' + str(self.topo)
        if self.rhoMult and not self.topo:
            self.beam_name += '_rho'

    def get_x_comp(self):

        diff_alpha_2 = (self.geometry.alpha() - self.geometry.alpha_bar) ** 2
        denominator = (1 - self.geometry.alpha_0) ** 2

        alpha_fact = diff_alpha_2 / denominator

        g_alpha = (self.geometry.aperture() / (np.pi * np.sqrt(self.geometry.alpha()) * (1 + self.geometry.alpha())) *
                   np.exp(-(self.sigma / 2) * alpha_fact))

        E_x = g_alpha
        Ph_x = self.topo * self.geometry.phi()

        if self.topo > 0 or self.rhoMult:
            E_x *= self.geometry.rho() / self.geometry.rho_max

        return E_x, Ph_x
