# David Maluenda Niubó - FAO (UB) [2014]
#
# [Ex,Ey,Px,Py] = beam_design(SLMsize=[1024 768],BeamType=last,infile=0,draw=0,stokes=0,gauss_correction=0)
#
# Returns tha amplitude of X and Y components and the phase between both. 
# Beam_type is the label of the designed beam

# function[E_x,E_y,Ph_x,Ph_y]=beam_design(SLM_size,beam_type,infile,draw,stokes,gauss_correction)
# clear all

import os
import sys
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def beam_design(SLM_size, beam_type=None, infile=None, verbose=0, stokes=None,
                gauss_correction=None, NA=1., rho_max=None, **kwargs):
    # close all;

    # if ~exist('SLM_size','var'), clear all, SLM_size = [1024 768]; end
    # if ~exist('beam_type','var'), beam_type = 45; end
    # if ~exist('infile','var'), infile = 0; end
    # if ~exist('draw','var'), draw = 1; end
    # if ~exist('stokes','var'), stokes = 0; end
    # if ~exist('gauss_correction','var'), gauss_correction = 0; end

    x, y = np.meshgrid(range(-SLM_size[1] // 2, SLM_size[1] // 2, ),
                       range(SLM_size[0] // 2, -SLM_size[0] // 2, -1))

    phi = np.mod(np.arctan2(y, x), 2 * np.pi)  # [0 2pi]
    rho = np.sqrt(x ** 2 + y ** 2)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    rho_max = min(SLM_size) / 2 if rho_max is None else rho_max
    theta_0 = np.arcsin(NA) if NA else np.pi / 2
    rho_0 = rho_max / np.tan(theta_0)
    theta = np.arctan2(rho, rho_0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    aperture = sin_theta <= NA
    win_size = (np.count_nonzero(aperture[SLM_size[0] // 2, :]),
                np.count_nonzero(aperture[:, SLM_size[1] // 2]))

    # Charo's notation
    alpha = cos_theta
    alpha_0 = np.cos(theta_0)

    if verbose > 1:
        plt.figure()
        fig, ax = plt.subplots(4, 2)
        ax[0, 0].imshow(x, cmap='gray')
        ax[0, 1].imshow(y, cmap='gray')
        ax[1, 0].imshow(rho / rho_max, cmap='gray', vmin=0, vmax=1)
        ax[1, 1].imshow(phi, cmap='gray', vmin=0, vmax=np.pi * 2)
        ax[2, 0].imshow(theta, cmap='gray', vmin=0, vmax=np.pi / 2)
        ax[2, 1].imshow(cos_theta, cmap='gray', vmin=0, vmax=1)
        ax[3, 0].imshow(sin_theta, cmap='gray', vmin=0, vmax=np.pi / 2)
        ax[3, 1].imshow(sin_theta < NA, cmap='gray', vmin=0, vmax=1)
        plt.show()

    E_x = np.zeros(SLM_size)
    E_y = E_x
    Ph_x = E_x
    Ph_y = E_x

    beamNAME = str(beam_type)  # init
    modulation = kwargs.get('ModulationType', None)
    suffix_name = '_' + modulation if modulation is not None else ''

    if beam_type == 87:  # switch beam_type

        sigma = kwargs.get('sigma', 5)
        topo = kwargs.get('topo', 0)
        pol = kwargs.get('pol', 'linear')
        avoid_center = kwargs.get('avoidCenter', False)

        beamNAME += '_s' + str(sigma)
        if topo:
            beamNAME += '_m' + str(topo)
        if (avoid_center or pol == 'radial') and not topo:
            beamNAME += '_rho'

        alpha_bar = (alpha_0 + 1) * 0.5

        diff_alpha_2 = (alpha - alpha_bar) ** 2
        denominator = (1 - alpha_0) ** 2
        alpha_factor = diff_alpha_2 / denominator

        g_alpha = np.exp(-sigma/2 * alpha_factor)
        g_alpha /= np.sqrt(alpha)
        g_alpha /= 1 + alpha

        E_x = g_alpha
        Ph_x = topo * phi

        if pol == 'radial' or avoid_center or topo > 0:
            E_x *= rho / rho_max

        print('All elements are positive?', (E_x * np.exp(1j * Ph_x) >= 0).all())
    #      case 86
    #
    #
    #         N    = 8;
    #         m    = 8;
    #         As   = NaN;
    #         NA   = 0.6;
    #         rhoM = 165/2;
    #
    # beamNAME=['Vec.Needle N=' num2str(N) ' m=' num2str(m) ' As=' num2str(As)];
    #
    #
    #         mask = (rho<=rhoM).*1;
    #         teta = asin(rho/rhoM*NA).*mask;
    #
    #         alpha = cos(teta);
    #         alpha0= cos(max(teta(:)));
    #         alpha1= cos(min(teta(:)));
    #         alphaB= m/2/N*(1-alpha0) + alpha0 ;
    #
    #         h = N*sinc( 2*N * (alpha-alphaB)./(alpha1-alpha0) ) .*mask ;
    #
    #         h = h/max(h(:));
    #
    #         Ex = h.*cos(phi);
    #         Ey = h.*sin(phi);
    #
    #         E_x  = abs(Ex);
    #         Ph_x = angle(Ex);
    #         E_y  = abs(Ey);
    #         Ph_y = angle(Ey);
    #
    #
    #     otherwise
    else:  # The basic ones
        if beam_type == 1:  # Radial
            beamNAME = 'Radial'
            k = 1
            l = 0
            theta_amp = 0
            theta_ph = 0
        elif beam_type == 2:  # Azimuthal
            beamNAME = 'Azimuthal'
            k = 1
            l = 0
            theta_amp = -np.pi / 2
            theta_ph = 0
        elif beam_type == 3:  # Star-like
            beamNAME = 'Star-like'
            k = 4
            l = 0
            theta_amp = 0
            theta_ph = 0
        else:
            beamNAME = 'Horizontal-Homogenous'
            k = 0
            l = 0
            theta_amp = 0
            theta_ph = 0

        E_x = np.cos(phi * k + theta_amp)
        E_y = np.sin(phi * k + theta_amp)
        Ph_x = np.angle(E_x)
        Ph_y = np.angle(E_y)
        #     p=find(xor(E_x<0,E_y<0));
        E_x = abs(E_x)
        E_y = abs(E_y)
    #     Ph=l*phi+theta_ph;
    #     Ph(p)=Ph(p)+pi;

    # if gauss_correction~=0
    #     R=5;
    #     f0=1;
    #     igauss=1.05-exp(-1/f0^2*(x.^2+y.^2).^1/R^2)*0.1;
    #     # figure;imagesc(igauss,[0 1])
    #     E_x=E_x.*igauss;
    #     E_y=E_y.*igauss;
    # end

    # Final adjustments
    E_x *= aperture
    E_x /= (E_x.max() + np.finfo(E_x.dtype).eps)
    E_y *= aperture
    E_y /= (E_y.max() + np.finfo(E_y.dtype).eps)
    Ph_x = np.mod(Ph_x, 2 * np.pi) * aperture
    Ph_y = np.mod(Ph_y, 2 * np.pi) * aperture

    # PATH=cd;
    # designPATH=[PATH '\Designs\design'];
    #
    # if infile~=0 #ES NORMALITZEN LES IMATGES!!
    #     #imagesc(E_x');
    #     disp(['beam_type=' beamNAME]);
    #     fNAME=[designPATH num2str(beam_type) '_Ex'];
    #     dlmwrite([fNAME '.dat'],E_x);
    #     imwrite(scripts.normalize_2D(E_x),[fNAME '.png'],'png')
    #     fNAME=[designPATH num2str(beam_type) '_Phx'];
    #     dlmwrite([fNAME '.dat'],Ph_x);
    #     imwrite(scripts.normalize_2D(Ph_x),[fNAME '.png'],'png')
    #     fNAME=[designPATH num2str(beam_type) '_Ey'];
    #     dlmwrite([fNAME '.dat'],E_y);
    #     imwrite(scripts.normalize_2D(E_y),[fNAME '.png'],'png')
    #     fNAME=[designPATH num2str(beam_type) '_Phy'];
    #     dlmwrite([fNAME '.dat'],Ph_y);
    #     imwrite(scripts.normalize_2D(Ph_y),[fNAME '.png'],'png')
    # end
    if verbose:
        I = (abs(E_x)) ** 2 + (abs(E_y)) ** 2
        plt.figure()
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].imshow(I, vmin=0, vmax=1)
        axs[0, 0].set_title('Design: I')
        axs[1, 0].imshow(E_x, vmin=0, vmax=1)
        axs[1, 0].set_title('Design: Ex')
        axs[1, 1].imshow(E_y, vmin=0, vmax=1)
        axs[1, 1].set_title('Design: Ey')
        axs[2, 0].imshow(Ph_x, vmin=0, vmax=2 * np.pi)
        axs[2, 0].set_title('Design: Ph_x')
        axs[2, 1].imshow(Ph_y, vmin=0, vmax=2 * np.pi)
        axs[2, 1].set_title('Design: Ph_y')
        Ph = np.mod(Ph_y - Ph_x, 2 * np.pi)
        axs[0, 1].imshow(Ph, vmin=0, vmax=2 * np.pi)
        axs[0, 1].set_title('Design: Ph')
        plt.show()
    #
    # fid = fopen('+scripts/design_kinds.txt');
    # tline = fgetl(fid);
    # beams_in_file=0;
    # while ischar(tline)
    #     tline = fgetl(fid);
    #     beams_in_file=beams_in_file+1;
    # end
    # fclose(fid);
    # if beam_type>beams_in_file #adds the name in the list 'beam_kinds.txt'
    #     dlmwrite('+scripts/design_kinds.txt',[num2str(beam_type) ': ' beamNAME], '-append','delimiter','','newline','pc')
    # else
    #     disp('Designing the beam:');
    #     fid=fopen('+scripts/design_kinds.txt');
    #     beamN=textscan(fid, '#s', 1,'delimiter',',','headerlines',beam_type-1);
    #     disp(beamN{1});
    # end
    #
    #
    # if stokes~=0
    #     [s0,s1,s2,s3,DOP]=scripts.stokes(E_x.*exp(1i*Ph_x),E_y.*exp(1i*Ph_y));
    #     figure
    #     imagesc(s0',[-1 1]);title 'S0'
    #     figure
    #     imagesc(s1',[-1 1]);title 'S1'
    #     figure
    #     imagesc(s2',[-1 1]);title 'S2'
    #     figure
    #     imagesc(s3',[-1 1]);title 'S3'
    #     figure
    #     imagesc(DOP',[-1 1]);title 'DOP'
    # end
    #
    # clear SLM_size

    return E_x, E_y, Ph_x, Ph_y, beamNAME + suffix_name


if __name__ == "__main__":
    print(f"WARNING: running {os.path.basename(sys.argv[0])} "
          f"as script should be just for testing.")

    E_x, E_y, Ph_x, Ph_y, name = beam_design((768 // 2, 1024 // 2), beam_type=87,
                                             verbose=2, NA=0.75, rho_max=50,
                                             sigma=50, topo=0)
