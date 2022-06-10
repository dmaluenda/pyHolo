



def utils_main(**kwargs):
    print(kwargs)


class Argument:

    def __init__(self, *args, **kwargs):
        self.labels = args
        self.required = kwargs.pop('required', False)
        self.kwargs = kwargs

    def get_name(self):
        return self.labels[0].strip('--')

    def is_required(self):
        return self.required


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

    return parser.parse_args(), extra_args


def get_and_check_mode_arguments(parser, mode_args):
    all_args, extra_args = get_all_arguments(parser)
    mode_args_names = [arg.get_name() for arg in mode_args] + extra_args
    all_args_dict = vars(all_args)

    mode_args_dict = {arg_name: all_args_dict.get(arg_name) for arg_name
                      in all_args_dict if arg_name in mode_args_names
                      and all_args_dict.get(arg_name) is not None}

    missing = []
    required = [arg.get_name() for arg in mode_args if arg.is_required()]
    for requirement in required:
        if not all_args_dict.get(requirement):
            missing.append(requirement)

    if missing:
        miss_str = "', '".join(missing)
        # parser.print_help()
        parser.error(f"'{miss_str}' arg_name(s) is/are required for "
                     f"'{all_args_dict.get('mode')}' mode.")

    return mode_args_dict
