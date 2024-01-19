#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  15  2021

@author: dmaluenda@ub.edu
"""

import argparse

from pyHolo.utils import Argument, get_all_arguments, get_mode_arguments, utils_main

from pyHolo.beam_simulation import beam_sim_main
from pyHolo.slm_calibration.calibration import calibrate_main
from pyHolo.holo_generator import holo_gen_main
# from wx_gui import wxMain as gui_production_main


# import pyHolo

""" This main program is just to manage the arguments in different modes.
        
        'modes' is the main dict with all main modes. The key is the mode name
         whereas the value is another dict which must contain 'help', 'func' and 'args'
            'help': just a mode description. It will be printed in help as:  key -> help
            'func': the main function which will be called (should be imported here above)
            'arg' : list of 'Argument' objects. 'Argument' objects are initialized as
                    in parser.add_argument(*args, **kwargs) function
"""

join = "\n\t".join(['afsaf','sdgsdghs','adagdgasdg'])
modes = {  # dict of main modes. Every mode must be a dict with 'help', 'func' and 'args'
         'gui-production': {
             'help': "GUI to show holograms and capture from cameras",
             'func': holo_gen_main,  # gui_production_main,
             'args': []
         },
         'beam_simulation': {
             'help': "Beam simulation",
             'func': beam_sim_main,
             'args': [Argument('--gui', action='store',
                               help='Numerical Aperture of the system'),]
         },
         'slm-cal': {  # The key mode is its name.
             'help': 'SLM calibration mode',
             'func': calibrate_main,
             'args': [Argument('--label', action='store', required=True,
                               help='Label used to read and write info'),
                      Argument('--SLM', action='store', metavar='1/2/all',
                               default=1,
                               help="Set it to 1, 2 or 'all' (1 is default)"),
                      Argument('--POLs', action='store', metavar='45/135/all',
                               type=str, default='all',
                               help="Set it to 45, 135 or 'all' "
                                    "(all is default)"),
                      Argument('--check_ROIs', action='store_true',
                               help='Display the stored ROIs to check '
                                    'if they are OK.'),
                      Argument('--ROIs_points', action='store', nargs=5,
                               metavar='',
                               help='5 coordinates corresponding to Y1, Y2, '
                                    'Y3, Xi and Xf. Y4 is calculated using '
                                    'the rest.'),
                      Argument('--check_peaks', action='store_true',
                               help='Display the FFT to check where is the peak.'),
                      Argument('--freq_peaks', action='store', nargs=2,
                               type=int,
                               help='The two frequency peaks for 45 and 135 degrees.'),
                      Argument('--ignore_amp', action='store_false',
                              help='To ignore the amplitude calibration.'),
                      Argument('--use_whole', action='store_true',
                             help='To ignore ROIs in the amplitude calibration.'),
                      Argument('--only_raw', action='store_true',
                               help='Only estimates the raw response. '
                                    'For first attempts...'),
                      Argument('--use_raw_pkl', action='store_true',
                                help='To avoid the raw estimation and use a '
                                     'precalculated data in the '
                                     'SLM_calibrations/<label>_raw_response.pkl '
                                     'file.')
                      ]
                     },
         'holo-gen': {
             'help': 'Hologram generator mode',
             'func': holo_gen_main,
             'args': [Argument('--beam_type', metavar='N',  # type=int,
                               help='bar help', required=True),
                      Argument('--NA', type=float, required=True,
                               help='Numerical Aperture of the system'),
                      Argument('--EP_edges', action='store', nargs=4,
                               metavar='',
                               help='4 coordinates: Xi, Xf, Yi and Yf of the EP '
                                    '(in pixels), corresponding to the '
                                    'positions where light starts getting in '
                                    'the MO (use a knife edge holo).'),
                      Argument('--rho_max', type=float,
                               help='Radius of the Entrance Pupil on the '
                                    'SLM plane in pixels. '
                                    'It is redundant with --EP_edges'),
                      ]
                     },
         'utils': {
             'help': 'Some useful tools',
             'func': utils_main,
             'args': [Argument('-s', required=True, metavar='SCRIPT',  # TODO: add a list of scripts
                               help=f'Run the SCRIPT file. {join}'
                                    f'Add --man to check some help.'),
                      Argument('--man', action='store_true',
                               help='To show some help for a certain script. '
                                    'Check some typical parameters below:'),
                      Argument('--filename', help="Main filename to work."),
                      Argument('--save', nargs='?', metavar='FILENAME', # type=str,
                               help="Save the result in 'FILENAME', if not "
                                    "passed it OVERWRITES the original. "
                                    "If it starts or ends with an underscore, "
                                    "it is understood as a suffix."),
                      # Argument('--stokes', action='store_true',
                      #          help="Computes the Stokes images. It is computed "
                      #               "from 6 images: PREFIX_aXX.png where XX is "
                      #               "0, 45, 90, 135, Dex are Lev. "
                      #               "If a whole FILENAME is past, the same filename "
                      #               "structure is expected."),
                      Argument('--crop', nargs='3,4', type=int, metavar='C',  # TODO: arrange metavar
                               help='Crops images around Cx,Cy center and with a '
                                    'size of Sx and Sy for a rectangular cropping. '
                                    'If just one size is passed then Sy=Sx.'),]
                   }
         }

parser = argparse.ArgumentParser(epilog="Any additional argument is passed "
                                        "to the final worker. "
                                        "Use: '--extraArg someValue'.")
parser.add_argument("mode", help="Program main mode. See groups below.",
                    choices=[m for m in modes])
parser.add_argument('--verbose', '-v', action='count', default=0,
                    help="To show/print more info. Use '-vv' to show even more.")

for mode_name, mode_dict in modes.items():
    group = parser.add_argument_group(f"{mode_name} -> {mode_dict['help']}")
    for mode_args in mode_dict['args']:
        group.add_argument(*mode_args.labels, **mode_args.kwargs)

all_args, extra_args = get_all_arguments(parser)
parsed_mode = all_args.get('mode')

mode_dict = modes.get(parsed_mode)

mode_kwargs, error = get_mode_arguments(mode_dict['args'], all_args, extra_args)
if error:
    # parser.print_help()
    parser.error(error+' Check --help for more details.')
else:
    # print(mode_dict)
    # print(mode_kwargs)
    mode_dict['func'](**mode_kwargs)