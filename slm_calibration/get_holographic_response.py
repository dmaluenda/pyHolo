#Complex values. (2012)
#David Maluenda Niubó - Applied Physics and Optics (UB)

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# label used to read and write info
label = '211209_lowFreq'
root = Path(os.getcwd()).parent / 'SLM_calibrations'


#k=2
# date=clock
N1 = 165
N2 = 175
A_max = 1  #just for ploting


# usefull values and SemiCercle

phi1_0 = 70 * np.pi / 180  # <-Rotate
phi2_0 = 60 * np.pi / 180  # <-Rotate
A1_maxCM = 0.30   # <-Trim1 in Complex Modulation
A2_maxCM = 0.25   # <-Trim2 in Complex Modulation
A1_maxRM = 0.55   # <-Trim1 in Real Modulation
A1_maxAM = 0.85   # <-Trim1 in Amplitude Modulation  (strictly positive)
phi1_0a = -45 * np.pi / 180  # <-Additional rotation for amplitude only
A2_maxAM = 0.5    # <-Trim2 in Amplitude Modulation

errA = 0.1
errP = 10/360

raw_response_fn = root / (label+'_raw_response.pkl')
with open(raw_response_fn, 'rb') as f:
    rawdata1 = pickle.load(f)


def smoother(data, fig=None, cnt=0):
    newdata = np.array(data)
    if fig is None:
        fig = plt.figure()
        plt.plot(newdata)
        plt.show()

    for idx in range(1, data.size-1):
        newdata[idx] = np.mean([data[idx-1], data[idx-1], data[idx],
                                data[idx+1], data[idx+1]])

    plt.figure()
    plt.plot(newdata)
    plt.show()
    if input("It is to much? (y/N): ").lower().startswith('y'):
        return data
    else:
        return smoother(newdata, fig, cnt+1)


def getResponse(slm, ending):
    ampli = rawdata1.get(f'A{slm}', None)
    phase = rawdata1.get(f'phi{slm}', None)

    if ampli is not None and phase is not None:
        ampli = smoother(ampli)
        phase = smoother(phase)

        phase *= np.pi / 180
        return ampli[:ending] * np.exp(1j*phase[:ending])

A1 = getResponse(1, N1)
A2 = getResponse(2, N2)


def getAccessiblesComplex(cArray, ending, cMax):
    accValue = []
    p_inset = []
    aux_i = []
    aux_j = []

    count = 0
    for i in range(ending):
        for j in range(i, ending):
            aux = (cArray[i] + cArray[j]) / 2
            accValue.append(aux)
            aux_i.append(i)
            aux_j.append(j)
            if abs(aux) <= cMax:
                p_inset.append(count)
            count += 1

    return np.array(accValue), np.array(p_inset), np.array(aux_i), np.array(aux_j)


def getAccessiblesAmplitude(allAccessible, phi0=0., A_max=1., allowNeg=True):
    # Amplitude only modulation
    N_At = 300  # maximum resolution (walking over the red line)
    A_min = -A_max if allowNeg else 0.
    Ats = np.linspace(A_min, A_max, N_At)
    aux1 = 0
    pA = []

    for iA in range(N_At):

        At1 = Ats[iA] * np.exp(1j * phi0)
        indx = np.argmin(abs(At1 - allAccessible))

        if indx != aux1:
            pA.append(indx)
            aux1 = indx

    return np.array(pA)


if A1 is not None:
    B1, pC1, aux1_i, aux1_j = getAccessiblesComplex(A1, N1, A1_maxCM)
    pR1 = getAccessiblesAmplitude(B1, phi1_0, A1_maxRM)
    pA1 = getAccessiblesAmplitude(B1, phi1_0+phi1_0a, A1_maxAM, allowNeg=False)

if A2 is not None:
    B2, pC2, aux2_i, aux2_j = getAccessiblesComplex(A2, N2, A2_maxCM)
    pR2 = getAccessiblesAmplitude(B2, phi2_0, A2_maxAM)
    pA2 = getAccessiblesAmplitude(B2, phi2_0, A2_maxAM, allowNeg=False)


def plotAndSave(filename, response, accessible, pointer1, pointer2,
                pComplex, pReal, pAmplitude, phi0=0., phi0a=0.,
                ACmax=1., ARmax=1., AMmax=1., **kwargs):

    data2store = {'help': "Write some help"}

    def create_data(amplitude, phase, indices, Amax=1.):
        keys = [(int(idx_i), int(idx_j)) for idx_i, idx_j in
                zip(pointer1[indices], pointer2[indices])]
        vals = [(amp / Amax, ph) for amp, ph in zip(amplitude, phase)]
        return {k: v for k, v in zip(keys, vals)}

    resp_abs = abs(response)
    resp_ph = np.angle(response) - phi0
    access_ph = np.angle(accessible) - phi0
    access_abs = abs(accessible)
    comp_ph = np.angle(accessible[pComplex]) - phi0
    comp_abs = abs(accessible[pComplex])
    real_ph = np.angle(accessible[pReal]) - phi0
    real_abs = abs(accessible[pReal])
    amp_ph = np.angle(accessible[pAmplitude]) - phi0
    amp_abs = abs(accessible[pAmplitude])

    dict_comp = create_data(comp_abs, comp_ph, pComplex, ACmax)
    data2store.update(complex_modulation=dict_comp)

    dict_real = create_data(real_abs, real_ph, pReal, ARmax)
    data2store.update(real_modulation=dict_real)

    dict_amplitude = create_data(amp_abs, amp_ph-phi0a, pAmplitude, AMmax)
    data2store.update(amplitude_modulation=dict_amplitude)

    # plotting
    plt.figure()
    plt.polar(access_ph, access_abs, '+g', label='Possible coding values')
    plt.polar(resp_ph, resp_abs, 'sc', markersize=3, label='SLM curve')
    plt.polar(comp_ph, comp_abs, '+b', label='Complex Modulation')
    plt.polar(real_ph, real_abs, 'xk', label='Real modulation',
              markersize=10)
    plt.polar(amp_ph, amp_abs, '+k', label='Amplitude only modulation',
              markersize=10)
    plt.polar(np.linspace(0, 2*np.pi, 100), [ACmax]*100, '-k', linewidth=3,
              label='complex modulation inset')
    plt.polar([0, np.pi] * 50, np.linspace(0, ARmax, 100), '-r', linewidth=3,
              label='real modulation inset')
    plt.polar([phi0a] * 50, np.linspace(0, AMmax, 50), '-y', linewidth=3,
              label='amplitude modulation inset')
    plt.title('SLM1')
    plt.legend()
    plt.show()

    with open(filename, 'wb') as file:
        pickle.dump(data2store, file)


    # cd plots
    # print(h,'-depsc',[num2str(date(3),'#02.0f') num2str(date(2),'#02.0f') ...
    #     num2str(date(1)-2000) '_polar_SLM1.eps'])
    # cd ..

    # # Creating data for SLM1
    # Complex Modulation
    # scored_vars = {}
    # scored_vars.update(phi=np.mod(access_ph-phi0, 2*np.pi))
    # dataCM1 = [ abs(accessible(pComplex))/A1_maxCM ; phi1(pComplex) ; aux1_i(pComplex) ; aux1_j(pComplex) ]
    # fid     = fopen( 'ComplexValues_SLM1.txt' , 'wt' )
    # fprintf( fid , '#10.20f  #10.20f  #3.0f  #3.0f\n' , dataCM1 )
    # fclose(fid)
    # # Amplitude Modulation
    # dataAM1 = [ abs(accessible(pAmplitude))/A1_maxAM ; phi1(pAmplitude) ; aux1_i(pAmplitude) ; aux1_j(pAmplitude) ]
    # fid     = fopen( 'AmplitudeValues_SLM1.txt' , 'wt' )
    # fprintf( fid , '#10.20f  #10.20f  #3.0f  #3.0f\n' , dataAM1 )
    # fclose(fid)



if A1 is not None:
    fn1 = root / (label+'_map_SLM1.pkl')
    plotAndSave(fn1, A1, B1, aux1_i, aux1_j, pC1, pR1, pA1,
                phi0=phi1_0, phi0a=phi1_0a,
                ACmax=A1_maxCM, ARmax=A1_maxRM, AMmax=A1_maxAM)

    # with open(fn1, 'rb') as file:
    #     data = pickle.load(file)
    #
    # print(data.get('help', None))

# #ploting SLM2
# h = figure
# polar( angle(A2) , abs(A2) , '-r' ) #Real curve
# hold on
# polar( angle(B2) , abs(B2) , '+g' ) #Possibles values
# polar( angle(B2(pC2)) , abs(B2(pC2)) , '+b' ) #Complex Modulation
# hp=polar(angle(B2(pA2)),abs(B2(pA2)),'ok')
# set(hp,'MarkerSize',10) # Amplitude Mod
# title 'SLM2'
# legend 'SLM curve' 'Possible coding values' 'Complex Modulation' 'Amplitude only modulation'
# hold off
# cd plots
# print(h,'-depsc',[num2str(date(3),'#02.0f') num2str(date(2),'#02.0f') ...
#     num2str(date(1)-2000) '_polar_SLM2.eps'])
# cd ..
#
#
# # # # Creating data for SLM2
# # Complex Modulation
# phi2    = mod(angle(B2)-phi2_0,2*pi)
# dataCM2 = [ abs(B2(pC2))/A2_maxCM ; phi2(pC2) ; aux2_i(pC2) ; aux2_j(pC2) ]
# fid     = fopen( 'ComplexValues_SLM2.txt' , 'wt' )
# fprintf( fid , '#10.20f  #10.20f  #3.0f  #3.0f\n' , dataCM2)
# fclose(fid)
# # Amplitude Modulation
# dataAM2 = [ abs(B2(pA2))/A2_maxAM ; phi2(pA2) ; aux2_i(pA2) ; aux2_j(pA2) ]
# fid     = fopen( 'AmplitudeValues_SLM2.txt' , 'wt' )
# fprintf( fid , '#10.20f  #10.20f  #3.0f  #3.0f\n' , dataAM2 )
# fclose(fid)
#
#
#

