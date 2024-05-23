
import numpy as np
from scipy import special

from mapa_holo import mapa_holo


def get_hologram(code, slm_size=(768, 1024), rho_max=None, NA=0.75,
                 modulation_type='_complex', calibration_file=None):
    modulation_type = modulation_type.lstrip('_')
    slm_size = np.array(slm_size) // 2  # Assuming that Arrizon uses 2x2 macropixel

    x, y = np.meshgrid(range(-slm_size[1] // 2,  slm_size[1] // 2),
                       range( slm_size[0] // 2, -slm_size[0] // 2, -1))

    phi = np.mod(np.arctan2(y, x), 2 * np.pi)  # [0 2pi]
    rho = np.sqrt(x ** 2 + y ** 2)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    rho_max = min(slm_size) / 2 if rho_max is None else int(rho_max)
    theta_0 = np.arcsin(NA) if NA else np.pi / 2
    rho_0 = rho_max / np.tan(theta_0)
    theta = np.arctan2(rho, rho_0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    aperture = sin_theta <= NA
    win_size = (np.count_nonzero(aperture[slm_size[0] // 2, :]),
                np.count_nonzero(aperture[:, slm_size[1] // 2]))

    # Charo's notation
    alpha = cos_theta
    alpha_0 = np.cos(theta_0)

    field = np.zeros_like(aperture, dtype='complex64')*5
    error = ''
    try:
        ldict = locals()
        exec(code, globals(), ldict)  # it has to generate a complex 'field'
        field = ldict['field']
    except Exception as e:
        error = "Some error occurred during the code execution.\n\n"
        error += str(e)
        return field.astype(np.uint8), error

    field *= aperture

    amplitude = np.abs(field)
    phase = np.angle(field)

    arrizon = mapa_holo(amplitude / amplitude.max(), phase, modulation_type, algorithm=3,
                       calibration_file=calibration_file).astype(np.uint16)
    direct = np.angle(field)#(np.mod(np.angle(field), 2*np.pi)/2/np.pi).astype(np.uint16)
    direct -= direct.min()
    direct /= direct.max()
    direct = (direct*2**10).astype(np.uint16)

    return direct, error


def mono2rgb_hologram(holo, slm_brand='HoloEye', true_monitor=True, force=False):
    r_ch = np.zeros_like(holo, dtype='uint8')
    g_ch = np.zeros_like(holo, dtype='uint8')
    b_ch = np.zeros_like(holo, dtype='uint8')

    def check_bitness(bitness):
        if holo.max() >= 2 ** bitness and not force:
            raise Exception(f"Trying to convert a non {bitness}-bit hologram "
                            f"to a {slm_brand} hologram. Set 'force=True' "
                            f"to keep going. However, it may loose "
                            f"information.")

    if slm_brand == 'HoloEye':
        check_bitness(8)

        r_ch = holo.astype('uint8')
        g_ch = r_ch.copy()
        b_ch = r_ch.copy()
        holo8 = r_ch.copy()

    elif slm_brand.startswith('SanTec'):
        check_bitness(10)

        kernel = np.linspace(0, 240, 16, dtype='uint8')
        r_lut = np.array(kernel.tolist() * 64, dtype='uint8')
        kernel2 = get_kernel(kernel, 8)
        g_lut = np.array(kernel2*8, dtype='uint8')
        kernel3 = get_kernel(kernel, 128)
        b_lut = np.array(kernel3, dtype='uint8')

        r_ch = r_lut[holo]
        g_ch = g_lut[holo]
        b_ch = b_lut[holo]

        holo8 = (holo/2**2).astype('uint8')

    else:
        raise Exception(f"{slm_brand} not recognized as a valid SLM brand "
                        f"to convert the hologram from mono to rgb.")

    rgb2u32 = lambda r, g, b: r * 2**16 + g * 2**8 + b  #  RGB->U32

    true_hologram = rgb2u32(r_ch, g_ch, b_ch)

    monitor = true_hologram if true_monitor else rgb2u32(holo8, holo8, holo8)

    return true_hologram.astype('uint32'), monitor.astype('uint32')




def get_kernel(kernel, rep):
    output = []
    for x in kernel:
        output += [x] * rep
    return output






