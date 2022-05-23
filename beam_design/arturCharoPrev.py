#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:57:58 2019

@author: artur.carnicer
"""

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import scipy.misc as spm

NA = 0.9
theta_0 = np.arcsin(NA)
alpha_0 = np.cos(theta_0)
NPz = 1001 #aumentar a 10001 para evitar oscilaciones en q
NPa = 500
Longz = 10.
d = 2.
count = 0

Numsig = 2  # 40
alphas = np.linspace(alpha_0, 1, NPa)
zetas = np.linspace(-Longz, Longz, NPz)
sigmas = np.linspace(-Numsig, 0, 2 * Numsig + 1)


def Fdealpha(alpha):
    return np.exp(0.5 * sigma * (alpha - 0.5 * (1 + alpha_0))**2 / (1 - alpha_0)**2)


def FpdealphaAnalitica2(alpha):
    return np.abs(Fdealpha(alpha) * (alpha - (1 + alpha_0) / 2) * sigma / (1 - alpha_0)**2)**2


def Gdealpha(alpha):
    return (alpha - alpha_0) * (Fdealpha(1) - Fdealpha(alpha_0)) / \
           (1 - alpha_0) + Fdealpha(alpha_0)


def F1dealpha(alpha):
    return Fdealpha(alpha) - Gdealpha(alpha)


def Fdealpha2(alpha):
    return np.abs(Fdealpha(alpha))**2


def alphamed(alpha):
    return alpha * Fdealpha2(alpha)


def integraEzR(alpha, z):
    return Fdealpha(alpha) * np.cos(2 * np.pi * z * alpha)


def integraEzI(alpha, z):
    return - Fdealpha(alpha) * np.sin(2 * np.pi * z * alpha)


BeI02 = np.array([], dtype=np.double)
kL0 = np.array([], dtype=np.double)
q = np.array([], dtype=np.double)


for sigma in sigmas:
    E_z2 = np.array([], dtype=np.double)
    I0 = spi.quad(Fdealpha2, alpha_0, 1)[0]
    print(sigma)
    alphabarra = spi.quad(alphamed, alpha_0, 1)[0] / I0

    B = (1 - alphabarra) * Fdealpha2(1) - (alpha_0 - alphabarra) * Fdealpha2(alpha_0) - I0

    BeI02 = np.append(BeI02, (B / I0)**2)

    Q = (np.abs(Fdealpha(1) + Fdealpha(alpha_0)) +
         np.abs(Fdealpha(1) - Fdealpha(alpha_0))) / np.sqrt(I0)

    #par_a = ((np.sqrt(d * Q) * (1 - alpha_0)**.25 + np.sqrt(d * Q * np.sqrt(1 - alpha_0) + 4 * np.pi * np.sqrt(d))) ** 2) / 4 / np.pi

    derF1_alpha = np.gradient(F1dealpha(alphas), alphas)
    KLA = np.sqrt(spi.simps(np.abs(derF1_alpha)**2, alphas) / I0)
#    KLA = np.sqrt(spi.quad(FpdealphaAnalitica2, alpha_0, 1)[0] / I0)
    KLB = Q / np.sqrt(1 - alpha_0)# / np.sqrt(np.pi)
    kL0 = np.append(kL0, (KLA + KLB))
    L0 = (KLA + KLB) / 2 / np.pi

    for z in zetas:
        PR = spi.quad(integraEzR, alpha_0, 1, args=(z))[0]
        PI = spi.quad(integraEzI, alpha_0, 1, args=(z))[0]
        E_z2 = np.append(E_z2, (PR*PR + PI*PI))

    omegaMediosPixels = np.uint16(NPz * L0 / 2 / Longz)
#    q = np.append(q, spi.simps(E_z2[np.uint16(0.5 * NPz - omegaMediosPixels) :
#        np.uint16(0.5 * NPz + omegaMediosPixels)], zetas[np.uint16(0.5 * NPz - omegaMediosPixels) :
#        np.uint16(0.5 * NPz + omegaMediosPixels)]) / spi.simps(E_z2, zetas))
    deltaX=2 * Longz / NPz
    q = np.append(q, spi.simps(E_z2[np.uint16(0.5 * NPz - omegaMediosPixels) :
        np.uint16(0.5 * NPz + omegaMediosPixels)], dx = deltaX) / spi.simps(E_z2, dx=deltaX))
    count +=1

#%%
plt.interactive(True)

plt.figure(1, figsize=(16, 9))
plt.plot(sigmas, BeI02)
plt.ylim(0., 1.1)
plt.xlim(-Numsig, 0)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\sigma$', fontsize=16)
plt.ylabel(r'$|B/I_0|^2$', fontsize=16)
plt.savefig('BI2.png')

plt.figure(2, figsize=(16, 9))
plt.plot(sigmas, kL0)
plt.ylim(0, kL0.max())
plt.xlim(-Numsig, 0)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\sigma$', fontsize=16)
plt.ylabel(r'$kL_0$', fontsize=16)
plt.savefig('kL0.png')

plt.figure(3, figsize=(16, 9))
plt.plot(sigmas, q)
plt.ylim(0, 1)
plt.xlim(-Numsig, 0)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(r'$\sigma$', fontsize=16)
plt.ylabel(r'$q$', fontsize=16)
plt.savefig('q.png')

plt.figure(4, figsize=(16, 9))
for sigma in [-1, -5, -10, -20, -40]:
    E_z2 = np.array([], dtype=np.double)
    for z in zetas:
        PR = spi.quad(integraEzR, alpha_0, 1, args=(z))[0]
        PI = spi.quad(integraEzI, alpha_0, 1, args=(z))[0]
        E_z2 = np.append(E_z2, (PR*PR + PI*PI))
    plt.plot(zetas, E_z2 , label = '$\sigma =$' + str(sigma))
#plt.ylim(0, 1)
#plt.xlim(-Longz, Longz)
plt.xlim(-10, 10)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel(r'$z\quad (\lambda)$', fontsize=16)
plt.ylabel(r'$|E(z)|^2$', fontsize=16)
plt.savefig('Iz.png')


plt.figure(5, figsize=(16, 9))
for sigma in [-1, -5, -10, -20, -40]:
    E_z2 = np.array([], dtype=np.double)
    for z in zetas:
        PR = spi.quad(integraEzR, alpha_0, 1, args=(z))[0]
        PI = spi.quad(integraEzI, alpha_0, 1, args=(z))[0]
        E_z2 = np.append(E_z2, (PR*PR + PI*PI))
    plt.plot(zetas, E_z2 / E_z2.max() , label = '$\sigma =$' + str(sigma))
#plt.ylim(0, 1)
#plt.xlim(-Longz, Longz)
plt.xlim(-10, 10)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel(r'$z\quad (\lambda)$', fontsize=16)
plt.ylabel(r'$|E(z)|^2$', fontsize=16)
plt.savefig('Iz-nor.png')
#%%
plt.figure(6, figsize=(16, 9))
for sigma in [-1, -5, -10, -20, -40]:
    plt.plot(alphas, np.abs(Fdealpha(alphas))**2, label = '$\sigma =$' + str(sigma))
plt.ylim(0, 1)
plt.xlim(alphas.min(), 1)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$|F(\alpha))|^2$', fontsize=16)
plt.savefig('F(a)2.png')