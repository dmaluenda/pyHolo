

from importlib import import_module
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import requests

def utils_main(**kwargs):

    script = kwargs.get('s')
    if script:
        print(f"The '{script}' script has been triggered...")
        script_module = import_module('scripts.'+script)
        script_module.main(**kwargs)
        return

    cropping = kwargs.get('crop', None)
    prefix_fn = kwargs.get('filename', None)

    im = None
    if kwargs.get('stokes', False):
        from scripts.get_stokes import main as get_stokes
        get_stokes(**kwargs)
    elif cropping:
        from scripts.get_stokes import crop
        im = crop(prefix_fn, *cropping, **kwargs)
    elif prefix_fn:
        plt.imshow(imageio.imread(prefix_fn))
        plt.show()


    new_filename = kwargs.get('save', None)
    if im is not None and new_filename is not None:
        new_filename = new_filename if new_filename else prefix_fn
        # print(im)
        imageio.imsave(new_filename, im)
        # print(imageio.imread(new_filename))





class Argument:

    def __init__(self, *args, **kwargs):
        self.labels = args

        self.required = kwargs.pop('required', False)
        help = kwargs.get('help', '')
        help = help + ' [Required]' if self.required else help

        self.nargs = kwargs.pop('nargs', None)
        self.nargs_flag = False
        if type(self.nargs) == str and ',' in self.nargs:
            kwargs.update(nargs='*')
            self.nargs_flag = True
            help = help + f' [Number of arguments: {" or ".join(self.nargs.split(","))}]'
        elif self.nargs is not None:
            kwargs.update(nargs=self.nargs)
        else:
            pass

        kwargs.update(help=help)
        self.kwargs = kwargs

    def get_name(self):
        return self.labels[0].strip('--')

    def is_required(self):
        return self.required

    def check_nargs(self):
        return self.nargs_flag

    def get_nargs(self):
        nargs = self.nargs
        return [True] if nargs == '*' else [int(x) for x in nargs.split(',')]


def get_all_arguments(parser):
    """ Parses all arguments and makes a list of extra arguments
    """
    known, unknown = parser.parse_known_args()

    extra_args = ['verbose']  # verbose is always an extra arg
    for arg in unknown:
        if arg.startswith(("--", "-")):
            # you can pass any arguments to add_argument
            arg_name = arg.split('=')[0]
            parser.add_argument(arg_name)
            extra_args.append(arg_name.strip('--'))

    return vars(parser.parse_args()), extra_args


def get_mode_arguments(mode_args, all_args_dict, extra_args):

    mode_args_names = [arg.get_name() for arg in mode_args] + extra_args

    mode_args_dict = {arg_name: all_args_dict.get(arg_name) for arg_name
                      in all_args_dict if arg_name in mode_args_names
                      and all_args_dict.get(arg_name) is not None}

    missing = []
    required = [arg.get_name() for arg in mode_args if arg.is_required()]
    for requirement in required:
        if not all_args_dict.get(requirement):
            missing.append(requirement)

    missing_error = ""
    if missing:
        missing_str = "', '".join(missing)
        mode = all_args_dict.get('mode')
        missing_error = f"'{missing_str}' argument(s) is/are required for '{mode}' mode."

    wrong_argn = []
    nargs_check = [(arg.get_name(), arg.get_nargs())
                   for arg in mode_args if arg.check_nargs()]
    for check_tuple in nargs_check:
        arg_name = check_tuple[0]
        parsed_arg = all_args_dict.get(arg_name)
        if parsed_arg and len(parsed_arg) not in check_tuple[1]:
            wrong_argn.append(arg_name)

    argn_error = ""
    if wrong_argn:
        argn_str = "', '".join(wrong_argn)
        argn_error = f"Incorrect number of arguments for '{argn_str}'."

    return mode_args_dict, missing_error or argn_error

def download_from_github(dest_path, relative='.',
                         org='WavefrontEngUB', repo='pyHolo',
                         branch='main'):

    github = 'https://raw.githubusercontent.com'

    url = f"{github}/{org}/{repo}/{branch}/{relative}/{str(dest_path.name)}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()  # error si falla

    dest_path.write_bytes(response.content)