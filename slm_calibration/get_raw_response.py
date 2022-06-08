## Evaluate the complex modulation response via evaluating the phase modulation
## and the amplitude modulation.
##
## The phase modulation is evaluated via the shift of vertical-interferences
## fringe-pattern produced when different gray level is displayed on two
## certain ROIs according to
##
##       Xi                  Xf
##   Y1   --------------------
##       |      ROI_up        | vertical interferences are inside of ROI_up
##   Y2   --------------------
##  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fringes-pattern breaks here
##   Y3   --------------------
##       |      ROI_dw        | vertical interferences are inside of ROI_dw
##   Y4   --------------------
##
##
## To evaluate the amplitude modulation two options are considered. One is
## directly from the sum of the pixels values of each images. The other, is
## comparing the sum of the pixels value in each ROI -descrived above- for
## every image (i.e. gray level).
##

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import find_peaks


root = Path(os.getcwd()).parent / 'SLM_calibrations'


def roi_def(label, Y1, Y2, Y3, Xi=0, Xf=1023, check_ROI=False):
    # ROIs definition
    Y4 = Y3 + Y2 - Y1  # To ensure the same area Up and down

    if check_ROI:
        imROI = plt.imread(root / f"{label}_I" / 'I1_100.png')

        imROI[:, Xi] = imROI.max()
        imROI[:, Xf] = imROI.max()
        imROI[Y1, :] = imROI.max()
        imROI[Y2, :] = imROI.max()
        imROI[Y3, :] = imROI.max()
        imROI[Y4, :] = imROI.max()

        print("\nCheck ROIs in figure...")
        plt.imshow(imROI)
        plt.colorbar()
        plt.title("Check both ROIs and set new key points in terminal "
                  "after close this.")
        plt.show()
        out = input("\nSet new key points parameters if needed "
                    "(empty to keep those drawn)\n"
                    "  Format -> Y1 Y2 Y3 Xi Xf : ")

        try:
            Y1, Y2, Y3, Xi, Xf = out.split(' ')
            print("Using custom ROI:", Y1, Y2, Y3, Xi, Xf)
        except:
            print("Using default ROI:", Y1, Y2, Y3, Xi, Xf)
            pass
    else:
        print("Using default ROI:", Y1, Y2, Y3, Xi, Xf)

    return Xi, Xf, Y1, Y2, Y3, Y4


def get_amplitude(label, roi, SLM, whole, vars_store):

    intensity_path = root / f"{label}_I"
    Xi, Xf, Y1, Y2, Y3, Y4 = roi

    #%% Amplittude modulation
    I0   = [0, 0]  # whole energy for im0.png
    fact = [0, 0]  # correction factor when ROIs mode is used
    I1 = np.zeros(256)  # array with the intensity modulation
    I2 = np.zeros(256)  # array with the intensity modulation
    A1 = None
    A2 = None

    for j in SLM:

        im = plt.imread(intensity_path / f"I{j}_0.png")
        up = im[Y1:Y2, Xi:Xf]
        down = im[Y3:Y4, Xi:Xf]
        fact[j-1] = up.sum() / down.sum()
        I0[j-1] = im.sum()

        for i in range(256):

            filename = intensity_path / f"I{j}_{i}.png"
            im       = plt.imread(filename)

            if whole:
                up = I0[j-1]
                down = im.sum()
            else:
                up = (fact[j-1] * im[Y1:Y2, Xi:Xf]).sum()
                down = (im[Y3:Y4, Xi:Xf]).sum()

            if j == 1:

                I1[i] = down / up

                if i == 255:
                    # To flip ref to signal
                    I1 = 1 / I1

                    # from intensity to amplitude
                    A1 = np.sqrt(I1/I1.max())

                    # plot and save the info
                    h = plt.figure()
                    plt.plot(A1)
                    plt.axis([0, 255, 0, 1.1])
                    plt.title(label+'_amplitude_SLM1')
                    plt.show()

                    vars_store.update(A1=A1)

                    # cd plots
                    # print(h,'-depsc',[label 'amplitude_SLM1.eps'])
                    # cd ..

            else:

                I2[i] = down.sum() / up.sum()

                if i == 255:

                    # To flip ref to signal
                    # I2 = 1./I2

                    # from intensity to amplitude
                    A2 = np.sqrt(I2/I2.sum())

                    # plot and save the info
                    h = plt.figure()
                    plt.plot(A2)
                    plt.axis([0, 255, 0, 1.1])
                    plt.title(label+'_amplitude_SLM2')
                    plt.show()

                    vars_store.update(A2=A2)

                    # cd plots
                    # print(h,'-depsc',[label 'amplitude_SLM2.eps'])
                    # cd ..

    return A1, A2

def get_single_phase(slm, k_str, Pf_candidate, check_peak, phase_path, roi):
    Xi, Xf, Y1, Y2, Y3, Y4 = roi

    Q = 1

    N   = np.zeros(256, dtype='int')
    M   = np.zeros(256, dtype='int')
    P   = np.zeros(256, dtype='int')
    phi = np.zeros(256)
    Dx  = Xf-Xi

    delta_phi = np.zeros(256)
    FFTup  = np.zeros((Dx, 256), dtype='csingle')
    FFTdw = np.zeros((Dx, 256), dtype='csingle')


    if check_peak or not Pf_candidate:
        # plot FFT(im) to evaluate where is the peak

        im = plt.imread(phase_path / f"ph{slm}_30.png")
        down = np.mean(im[Y3:Y4, Xi:Xf], axis=0)

        plt.figure()
        plt.plot(abs(np.fft.fft(down))[0:60], 'x')
        plt.grid()
        plt.show()

        try:
            Pf = int(input(f'Type the frequency of the first peak: '
                           f'(default {Pf_candidate})\n  '))
        except:
            Pf = Pf_candidate

    else:

        Pf = Pf_candidate

    print(f"Frequency used for {k_str}: {Pf}")

    for i in range(256):
        filename = phase_path / f'ph{slm}_{i}.png'

        im = plt.imread(filename)

        up = np.mean(im[Y1:Y2, Xi:Xf], axis=0)
        down = np.mean(im[Y3:Y4, Xi:Xf], axis=0)

        UP = np.fft.fft(up)
        FFTup[:, i] = UP
        DW = np.fft.fft(down)
        FFTdw[:, i] = DW

        if Pf:
            UP[2:Pf - Q] = 0
            UP[Pf + Q:Dx - Pf + 2 - Q] = 0
            UP[Dx - Pf + 2 + Q:Dx] = 0

            DW[2:Pf - Q] = 0
            DW[Pf + Q:Dx - Pf + 2 - Q] = 0
            DW[Dx - Pf + 2 + Q:Dx] = 0

        up2 = abs(np.fft.ifft(UP))
        dw2 = abs(np.fft.ifft(DW))

        if i == 0:
            up_draw = up2 - min(up2)
            up_draw = up_draw / max(up_draw) * 2 - 1
            up_draw = up_draw * (Y1 / 2 - Y2 / 2) + Y1 / 2 + Y2 / 2

            dw_draw = dw2 - min(dw2)
            dw_draw = dw_draw / max(dw_draw) * 2 - 1
            dw_draw = dw_draw * (Y3 / 2 - Y4 / 2) + Y3 / 2 + Y4 / 2

            plt.figure()
            plt.imshow(im)
            plt.plot(range(Xi, Xf), up_draw, linewidth=3)
            plt.plot(range(Xi, Xf), dw_draw, linewidth=3)
            plt.title(f"ph{slm}_{i}_45.png")
            plt.show()

        peakUP, _ = find_peaks(up2)
        peakDW, _ = find_peaks(dw2)

        P[i] = peakUP[5] - peakUP[4]  # Period

        N[i] = peakUP.shape[0]
        M[i] = peakDW.shape[0]

        if N[i] > M[i]:
            D = peakUP[1:M[i]] - peakDW[1:M[i]]
        elif M[i] > N[i]:
            D = peakUP[1:N[i]] - peakDW[1:N[i]]
        else:
            D = peakUP - peakDW

        if slm == 1:  # --- The sign change decreasing to increasing phi(i)
            Dmean = np.mean(D)
        else:
            Dmean = -np.mean(D)

        phi[i] = 360 * Dmean / P[i]

        delta_phi[i] = np.std(D) * 360 / P[i]  # deviation of the mean

    # [phi_N,pFirst] = antimod(phi,360)
    pFirst = 50
    return np.mod(phi - min(phi[0:pFirst]), 360)



def get_phase(label, roi, SLM, POLs, freq_peaks, Q, check_peak, vars_store):
    #%% Phase modulation
    Xi, Xf, Y1, Y2, Y3, Y4 = roi
    N   = np.zeros(256, dtype='int')
    M   = np.zeros(256, dtype='int')
    P   = np.zeros(256, dtype='int')
    phi = np.zeros(256)
    Dx  = Xf-Xi

    delta_phi = np.zeros(256)
    FFTup  = np.zeros((Dx, 256), dtype='csingle')
    FFTdw = np.zeros((Dx, 256), dtype='csingle')

    phi1_45  = None
    phi1_135 = None
    phi2_45  = None
    phi2_135 = None

    freq_peaks = freq_peaks if freq_peaks else [None, None]

    for k_str in POLs:

        phase_path = root / f"{label}_{k_str}"
        Pf_candidate = freq_peaks[0] if k_str == '45' else freq_peaks[-1]

        for j in SLM:

            phi_N = get_single_phase(j, k_str, Pf_candidate, check_peak,
                                     phase_path, roi)

            # plt.plot(range(256), np.mod(phi_N, 360))
            # plt.show()

            if k_str == '45':
                if j == 1:  # SLM1
                    # phi_0 = min(phi(1:35))
                    # phi = phi-phi_0
                    # phi = mod(phi,360)
                    phi1_45 = phi_N
                    delta_phi1_45 = delta_phi
                else:  # SLM2
                    # phi_0 = min(phi(1:35))
                    # phi = phi-phi_0
                    # phi = mod(phi,360)
                    phi2_45 = phi_N
                    delta_phi2_45 = delta_phi
            else:  # P@135º
                if j == 1:  # SLM1
                    # phi_0 = min(phi(1:35))
                    # phi = phi-phi_0
                    # phi = mod(phi,360)
                    phi1_135 = phi_N
                    delta_phi1_135 = delta_phi
                else:  # SLM2
                    # phi_0 = min(phi(1:35))
                    # phi = phi-phi_0
                    # phi = mod(phi,360)
                    phi2_135 = phi_N
                    delta_phi2_135 = delta_phi

    phi1 = phase_avg(phi1_45, phi1_135)
    phi2 = phase_avg(phi2_45, phi2_135)

    # plot and save info
    if 1 in SLM:
        plt.figure()  # phase for SLM1
        plt.plot(range(256), np.mod(phi1_45, 360), ':b', label='phi1_45')
        plt.plot(range(256), np.mod(phi1_135, 360), ':r', label='phi1_135')
        plt.plot(range(256), phi1, '-k', label='phi1')
        plt.legend()
        plt.show()
        # cd plots
        # print(f"{label}_phase_SLM1.eps")
        # cd ..

        phi1 = phi1 - phi1.min()
        vars_store.update(phi1=phi1)

    if 2 in SLM:
        plt.figure()  # phase for SLM2
        plt.plot(range(256), np.mod(phi2_45, 360), ':b')
        plt.plot(range(256), np.mod(phi2_135, 360), ':r')
        plt.plot(range(256), phi2, '-k')
        plt.show()
        # cd plots
        # print(h,'-depsc',[label 'phase_SLM2.eps'])
        # cd ..

        phi2 = phi2 - phi2.min()
        vars_store.update(phi2=phi2)

    return phi1, phi2

def phase_avg(phase1, phase2):
    if phase1 is not None and phase2 is not None:
        return np.mod((phase1 + phase2) / 2, 360)
    elif phase1 is not None:
        return np.mod(phase1, 360)
    elif phase2 is not None:
        return np.mod(phase2, 360)
    else:
        return np.zeros(256)


def main(**kwargs):
    # label, check_ROI=False, check_peak=True, peaks=(15, 16),
    #      SLM=(1,), POLs=('45', '135'), whole=False, Ts=True):

    # default ROI key points:
    Y1 = 150
    Y2 = 290
    Y3 = 380
    Xi = 0
    Xf = 1023

    label = kwargs.get('label', '')
    SLM = ([kwargs.get('SLM', 1)])
    SLM = (1, 2) if SLM == 'all' else tuple(SLM)
    POLs = kwargs.get('POLs', 'all')
    POLs = ('45', '135') if POLs == 'all' else tuple(POLs)
    ROIs_points = kwargs.get('ROIs_points', None)
    if ROIs_points is not None:
        Y1, Y2, Y3, Xi, Xf = ROIs_points
    check_ROIs = kwargs.get('check_ROIs', ROIs_points is None)
    freq_peaks = kwargs.get('freq_peaks', None)
    check_peak = kwargs.get('check_peaks',freq_peaks is None)
    Q = kwargs.get('Q', 1)
    Ts = kwargs.get('ignore_amp', True)
    whole = kwargs.get('use_whole', False)


    vars_store = {}

    roi = roi_def(label, Y1=Y1, Y2=Y2, Y3=Y3, Xi=Xi, Xf=Xf, check_ROI=check_ROIs)

    if Ts:
        A1, A2 = get_amplitude(label, roi, SLM, whole, vars_store)

    if POLs:
        phi1, phi2 = get_phase(label, roi, SLM, POLs, freq_peaks, Q, check_peak, vars_store)

    # %% Plot and save polar info
    if Ts and POLs:

        if 1 in SLM:
            plt.figure()  # polar for SLM1
            plt.polar(phi1 * np.pi / 180, A1)
            plt.show()
            # cd plots
            # print(h,'-depsc',[label 'polar_SLM1.eps'])
            # cd ..

        if 2 in SLM:
            plt.figure()  # polar for SLM2
            plt.polar(phi2 * np.pi / 180, A2)
            plt.show()
            # cd plots
            # print(h,'-depsc',[label 'polar_SLM2.eps'])
            # cd ..

    raw_response_fn = root / (label + '_raw_response.pkl')
    print(f"This variables is stored in '{raw_response_fn}':")
    print(', '.join([("%s %s" % (k, v.shape)) for k, v in vars_store.items()]))
    with open(raw_response_fn, "wb") as file:
        pickle.dump(vars_store, file)

    # # To load the stored data
    # with open(raw_response_fn, 'rb') as f:
    #     newdict = pickle.load(f)
    #
    # print(newdict)

    return vars_store


if __name__ == '__main__':
    # %% Parameters

    # label used to read and write info
    label = '20220523'

    # SLMs to be determined
    SLM = (1,)  # can be (1, 2)

    # Positions of Pol: 1 = P@45º  ;  2 = P@135º
    POLs = ('45', '135')  # can be one, two or empty to avoid evaluation

    # to cheack the FFT peak
    check_peak = True  # set True to check the FFT peak
    Pfs = (16, 24)  # options: (a, b) or (a,)
    Q = 1

    # Transmitions to evaluate
    Ts = True  # set False to avoid evaluation

    # mode to determine the transmittion modulation response
    whole = False  # False: ROIs is used ; True: whole sum is used

    main(label, SLM=(1,), POLs=('45','135'), check_peak=True, Pfs=(16,24),
         Ts=True, whole=False)
