#Hologram generator
#grayL=f(Trans_1,Trans_2,Phase)
#[0 1]=f([0 1],[0 1],[0 2pi])
# function[SLM1,SLM2]=mapa_Holo(Trans1,Trans2,Phase1,Phase2,ModulationType)
# set(handles.Status,'String','Calculating the holograms, this may take some time. Take it easy!');

import os
import sys
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import find_peaks


# label used to read and write info
label = '211209_lowFreq'
root = Path(os.getcwd()).parent / 'SLM_calibrations'
filenameTMP = label+'_map_SLM%d.pkl'

def get_modulation(slm, ModulationType, verbose=0):
    with open(root / (filenameTMP % slm), 'rb') as file:
        data = pickle.load(file)

    map = data.get(ModulationType + '_modulation', None)

    T_SLM1 = np.array([tup[0] for tup in map.values()])
    ph_SLM1 = np.array([tup[1] for tup in map.values()])
    Mapa1_1 = np.array([tup[0] for tup in map.keys()], dtype='int')
    Mapa2_1 = np.array([tup[1] for tup in map.keys()], dtype='int')

    if verbose > 1:
        plt.figure()
        plt.polar(ph_SLM1, T_SLM1, 'x')
        plt.show()

    return T_SLM1 * np.exp(1j*ph_SLM1), Mapa1_1, Mapa2_1

def mapa_holo(Trans1, Phase1, ModulationType='complex', verbose=0, **kwargs):

    # macropixel=get(handles.Macropixel,'value')
    # Just for script to check it:
    # clear all;
    # [Trans1,Trans2,Phase1,Phase2]=scripts.beam_design([1024 768],1);

    if not ModulationType in ('amplitude', 'complex', 'real'):
        sys.exit('ModulationType must be either "Amplitude" or "Complex"')

    # mapping of accessible values
    C_SLM1, Mapa1_1, Mapa2_1 = get_modulation(1, ModulationType, verbose)

    #sizes
    N = Phase1.shape

    Amp_max1 = 1
    Amp_max2 = 1
    A_max    = min( [Amp_max1, Amp_max2] )

    # desirable values
    C1 = Trans1 * np.exp(1j*Phase1) * A_max

    if verbose > 1:
        print(C1.shape)

    # #---resizing-for-macropixel-procedure----------------
    # C1_mean(1:(Y(2)-Y(1)+1)/2,1:(X(2)-X(1)+1)/2) = ...
    #     (C1(Y(1):2:Y(2),X(1):2:X(2)) + ...
    #     C1(Y(1)+1:2:Y(2),X(1):2:X(2)) + ...
    #     C1(Y(1):2:Y(2),X(1)+1:2:X(2)) + ...
    #     C1(Y(1)+1:2:Y(2),X(1)+1:2:X(2)))/4 ;
    # C2_mean(1:(Y(2)-Y(1)+1)/2,1:(X(2)-X(1)+1)/2) = ...
    #     (C2(Y(1):2:Y(2),X(1):2:X(2)) + ...
    #     C2(Y(1)+1:2:Y(2),X(1):2:X(2)) + ...
    #     C2(Y(1):2:Y(2),X(1)+1:2:X(2)) + ...
    #     C2(Y(1)+1:2:Y(2),X(1)+1:2:X(2)))/4 ;
    #
    SLM1 = np.zeros(np.dot(N, 2), dtype='uint8')
    field = np.zeros(N, dtype='csingle')

    p1 = np.zeros(C1.shape, dtype='int')
    for i in range(N[0]):
       for j in range(N[1]):
            p1[i, j] = np.argmin(abs(C1[i, j] - C_SLM1))

            SLM1[2 * i, 2 * j] = Mapa1_1[p1[i, j]]
            SLM1[2 * i+1, 2 * j+1] = Mapa1_1[p1[i, j]]
            SLM1[2 * i, 2 * j+1] = Mapa2_1[p1[i, j]]
            SLM1[2 * i+1, 2 * j] = Mapa2_1[p1[i, j]]

            field[i, j] = C_SLM1[p1[i, j]]

    if verbose > 1:
        plt.figure()
        fig, axs = plt.subplots(2,1)
        axs[0].imshow(np.angle(field))
        axs[1].imshow(abs(field))
        plt.show()

    # print(p1.shape)

    # SLM1[::2, ::2] = Mapa1_1[p1[:, :]]
    # SLM1[1::2, 1::2] = Mapa1_1[p1[:, :]]
    # SLM1[1::2, ::2] = Mapa2_1[p1[:, :]]
    # SLM1[::2, 1::2] = Mapa2_1[p1[:, :]]




    # SLM1(Y(1)+1:2:Y(2),X(1)+1:2:X(2)) = ...
    #     Mapa1_1(p1(1:(Y(2)-Y(1)+1)/2,1:(X(2)-X(1)+1)/2)); # \
    # SLM1(Y(1)+1:2:Y(2),X(1):2:X(2)) = ...
    #     Mapa2_1(p1(1:(Y(2)-Y(1)+1)/2,1:(X(2)-X(1)+1)/2));
    # SLM1(Y(1):2:Y(2),X(1)+1:2:X(2))= ...
    #     Mapa2_1(p1(1:(Y(2)-Y(1)+1)/2,1:(X(2)-X(1)+1)/2)); # /
    #
    # SLM2(Y(1):2:Y(2),X(1):2:X(2)) = ...
    #     Mapa1_2(p2(1:(Y(2)-Y(1)+1)/2,1:(X(2)-X(1)+1)/2));
    # SLM2(Y(1)+1:2:Y(2),X(1)+1:2:X(2)) = ...
    #     Mapa1_2(p2(1:(Y(2)-Y(1)+1)/2,1:(X(2)-X(1)+1)/2)); # \
    # SLM2(Y(1)+1:2:Y(2),X(1):2:X(2)) = ...
    #     Mapa2_2(p2(1:(Y(2)-Y(1)+1)/2,1:(X(2)-X(1)+1)/2));
    # SLM2(Y(1):2:Y(2),X(1)+1:2:X(2)) = ...
    #     Mapa2_2(p2(1:(Y(2)-Y(1)+1)/2,1:(X(2)-X(1)+1)/2)); # /
    #
    #
    # SLM1 = (SLM1 - 1)/255
    #
    # [SLM1,SLM2]=scripts.rotate_SLM(SLM1,SLM2);
    # #
    plt.figure()
    plt.imshow(SLM1, cmap='gray')
    plt.show()
    # figure
    # # imshow(SLM2')
    # # figure
    # # imagesc(m1)
    # # figure
    # # imagesc(m2)

    return SLM1

def mapa_holo_plain(Trans, Phase, N, SLM_number=1, ModulationType='complex', **kwargs):

    C_SLM1, Mapa1_1, Mapa2_1 = get_modulation(SLM_number, ModulationType)

    C = Trans * np.exp(1j * Phase * np.pi / 180)  # Desirable value
    p = np.argmin(abs(C - C_SLM1))  # index of desirable value

    amp_ref = kwargs.get('amp_ref', 1)
    ph_ref = kwargs.get('ph_ref', np.pi)
    C0 = amp_ref * np.exp(1j * ph_ref)
    p0 = np.argmin(abs(C0 - C_SLM1))  # index of reference value


    SLM1 = np.zeros(N, dtype='uint8')

    if not kwargs.get('semi', False):
        SLM1[0::2, 0::2] = Mapa1_1[p]
        SLM1[1::2, 1::2] = Mapa1_1[p]
        SLM1[1::2, 0::2] = Mapa2_1[p]
        SLM1[0::2, 1::2] = Mapa2_1[p]
    else:
        # desirable value
        mid = N[0]//2  # this might change for SLM2
        SLM1[0:mid:2, 0::2] = Mapa1_1[p]
        SLM1[1:mid:2, 1::2] = Mapa1_1[p]
        SLM1[1:mid:2, 0::2] = Mapa2_1[p]
        SLM1[0:mid:2, 1::2] = Mapa2_1[p]

        # reference value
        SLM1[mid::2, 0::2] = Mapa1_1[p0]
        SLM1[mid+1::2, 1::2] = Mapa1_1[p0]
        SLM1[mid+1::2, 0::2] = Mapa2_1[p0]
        SLM1[mid::2, 1::2] = Mapa2_1[p0]

    return SLM1


if __name__ == '__main__':
    empty = np.random.random((1024, 768))
    mapa_holo(empty, empty)
