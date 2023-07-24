#Hologram generator
#grayL=f(Trans_1,Trans_2,Phase)
#[0 1]=f([0 1],[0 1],[0 2pi])
# function[SLM1,SLM2]=mapa_Holo(Trans1,Trans2,Phase1,Phase2,ModulationType)
# set(handles.Status,'String','Calculating the holograms, this may take some time. Take it easy!');

import os
import sys
import pickle
import json
import time
import numpy as np
import matplotlib.pyplot as plt

import pyopencl as cl

from pathlib import Path
from scipy.signal import find_peaks
from scipy.spatial import KDTree


# label used to read and write info
label = 'IDSfirst'  # '211209_lowFreq'
root = Path(os.getcwd())
if not (root / 'SLM_calibrations').is_dir():
    root = Path(__file__).parent.parent

filenameTMP = label+'_map_SLM%d.pkl'

def get_modulation(slm, ModulationType, verbose=0):
    with open(root / 'SLM_calibrations' / (filenameTMP % slm), 'rb') as file:
        data = pickle.load(file)

    map = data.get(ModulationType + '_modulation', None)

    T_SLM1 = np.array([tup[0] for tup in map.values()])
    ph_SLM1 = np.array([tup[1] for tup in map.values()])
    Mapa1_1 = np.array([tup[0] for tup in map.keys()], dtype='int')
    Mapa2_1 = np.array([tup[1] for tup in map.keys()], dtype='int')

    if verbose > 1:
        plt.figure()
        plt.polar(ph_SLM1, T_SLM1, 'x')
        plt.title('SLM%d' % slm)
        plt.show()

    return T_SLM1 * np.exp(1j*ph_SLM1), Mapa1_1, Mapa2_1

def mapa_holo_LV(holo_expression, Nx, Ny, rho_max, NA, ModulationTypeNum):
    
    Nx //= 2
    Ny //= 2
    rho_max //= 2

    # plt.figure()
    # plt.title(holo_expression)
    # plt.show()

    ModulationType = ('amplitude' if ModulationTypeNum==0 else 
                     ('real' if ModulationTypeNum==2 else 'complex'))

    x, y = np.meshgrid(range(-Nx//2,  Nx//2,),
                       range( Ny//2, -Ny//2, -1))

    phi = np.mod(np.arctan2(y, x), 2*np.pi)  # [0 2pi]
    rho = np.sqrt(x**2+y**2)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    rho_max = min(SLM_size)/2 if rho_max is None else rho_max
    theta_0 = np.arcsin(NA) if NA else np.pi/2
    rho_0 = rho_max / np.tan(theta_0)
    theta = np.arctan2(rho, rho_0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    aperture = sin_theta <= NA
    win_size = (np.count_nonzero(aperture[Ny//2, :]),
                np.count_nonzero(aperture[:, Nx//2]))
    rho /= rho_max

    # Charo's notation
    alpha = cos_theta
    alpha_0 = np.cos(theta_0)

    # compile(holo_expression)
    field = eval(holo_expression)

    field *= aperture

    amplitude = np.abs(field)
    phase = np.angle(field)

    # plt.figure()
    # plt.imshow(amplitude)
    # plt.show()

    return mapa_holo(amplitude, phase, ModulationType, algorithm=3).tolist()



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
    A_max = min( [Amp_max1, Amp_max2] )

    # desirable values
    C1 = Trans1 * np.exp(1j*Phase1)  * A_max

    if verbose > 1:
        print(C1.shape)

    algorithm = kwargs.get('algorithm')

    t0 = time.time()
    SLM1, field = get_holo(C1, C_SLM1, Mapa1_1, Mapa2_1, algorithm, verbose)
    t1 = time.time()

    # print(f'{algorithm}: Time to generate hologram: {t1-t0:.2f} s')

    if verbose > 1:
        fig, axs = plt.subplots(3,1)
        axs[0].imshow(np.angle(field))
        axs[0].set_title('Phase')
        axs[1].imshow(abs(field))
        axs[1].set_title('Amplitude')
        axs[2].imshow(SLM1, cmap='gray')
        axs[2].set_title('SLM1')
        plt.show()

    # #
    # figure
    # # imshow(SLM2')
    # # figure
    # # imagesc(m1)
    # # figure
    # # imagesc(m2)

    return SLM1


def get_holo(C1, C_SLM1, Mapa1_1, Mapa2_1, algorithm=0, verbose=0):

    if algorithm == 0:
        idx = get_holo_brute(C1, C_SLM1, verbose)
    elif algorithm == 1:
        idx = get_holo_npWhere(C1, C_SLM1, verbose)
    elif algorithm == 2:
        idx = get_holo_KDtree(C1, C_SLM1, verbose)
    elif algorithm == 3:
        idx = get_holo_openGL(C1, C_SLM1, verbose)
    else:
        sys.exit('Algorithm not implemented')

    SLM1 = np.zeros(np.dot(C1.shape, 2), dtype='uint8')
    SLM1[0::2, 0::2] = Mapa1_1[idx]
    SLM1[1::2, 1::2] = Mapa1_1[idx]
    SLM1[0::2, 1::2] = Mapa2_1[idx]
    SLM1[1::2, 0::2] = Mapa2_1[idx]

    if verbose > 1:
        plt.figure()
        plt.imshow(idx)
        plt.colorbar()
        plt.show()

    return SLM1, C_SLM1[idx]

def get_holo_openGL(C1, C_SLM1, verbose):

    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    N = C1.shape
    n = C1.size
    desired_flat = C1.flatten()
    desired_real = desired_flat.real.astype(np.float32)
    desired_imag = desired_flat.imag.astype(np.float32)

    if verbose>1:
        print(desired_flat.shape)
        print(np.reshape(desired_flat, N).shape)

    slm_flat = np.zeros_like(desired_real, dtype='float32')

    acc_real = C_SLM1.real.astype(np.float32)
    acc_imag = C_SLM1.imag.astype(np.float32)

    # plt.figure()
    # plt.plot(acc_real, acc_imag, ".b")

    m = acc_imag.size
    if verbose>1:
        print(f"m: {m}")


    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=my_gpu_devices)
    queue = cl.CommandQueue(ctx)

    # plt.figure()
    # plt.imshow(desired_real.reshape(N))

    mf = cl.mem_flags
    dr_buf = cl.Buffer\
       (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=desired_real)
    di_buf = cl.Buffer\
       (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=desired_imag)
    ar_buf = cl.Buffer\
       (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=acc_real)
    ai_buf = cl.Buffer\
       (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=acc_imag)
    slm_buf = cl.Buffer(ctx, mf.WRITE_ONLY, slm_flat.nbytes)

    with open(root / 'holo_generator' / 'mapa_holo_kernel.cl', 'r') as file:
        openCL_code = file.read()

    t0 = time.time()
    prg = cl.Program(ctx, openCL_code).build()
    t1 = time.time()

    prg.nearest(queue, slm_flat.shape, None,
                np.uint16(m), dr_buf, di_buf, ar_buf, ai_buf, slm_buf)

    t2 = time.time()
    res_slm = np.empty_like(slm_flat)
    t3 = time.time()
    cl.enqueue_copy(queue, res_slm, slm_buf)
    t4 = time.time()
    # print(" --- ")
    # print(f"cl.Program:  {t1-t0:.2f}")
    # print(f"prg.nearest: {t2-t1:.2f}")
    # print(f"np.empty_lk: {t3-t2:.2f}")
    # print(f"cl.enque_cp: {t4-t3:.2f}")

    # plt.figure()
    # plt.plot(res_slm)

    p1 = res_slm.reshape(N).astype('int')

    return p1


def get_holo_npWhere(C1, C_SLM1, verbose):
    N = C1.shape
    p1 = np.zeros(C1.shape, dtype='int')

    for i in range(N[0]):
        for j in range(N[1]):
            p1[i, j] = np.where(abs(C_SLM1-C1[i,j])==abs(C_SLM1-C1[i,j]).min())[0]

    return p1

def get_holo_KDtree(C1, C_SLM1, verbose):
    N = C1.shape

    data = np.array([C_SLM1.real, C_SLM1.imag]).T
    print(data.shape)
    tree = KDTree(data)
    p1 = np.zeros(C1.shape, dtype='int')
    for i in range(N[0]):
        for j in range(N[1]):
            point = np.array([C1[i, j].real, C1[i, j].imag])
            d, p1[i, j] = tree.query(point, workers=2)

    return p1

def get_holo_brute(C1, C_SLM1, verbose):

    N = C1.shape
    idx = np.zeros(C1.shape, dtype='int')  

    for i in range(N[0]):
        for j in range(N[1]):
            idx[i, j] = np.argmin(abs(C1[i, j] - C_SLM1))

    return idx


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
