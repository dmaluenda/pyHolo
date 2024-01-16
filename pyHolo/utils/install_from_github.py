



def install_devel_module(module_name, module_git=None, module_path=None, branch=None):
    """ This function is just to import a module which might be under developing.

        There are three options:
            1. The module is already installed in the system, so let's import it.
            2. The module is not installed, but it is located in the PC, so let's install it and import it.
            3. The module is not installed, and it is not located in the PC, so let's git-clone it, install it, and imported it.

        :param module_name: The module that we want to import
        :param module_path: The path where to find that module. If not passed, used ./<module_name>
        :param module_git: The https route where to find that module. If not passed, used .../WavefrontEngUB/<module_name>
        :param branch: Just if you want a certain branch of this repo.
        :return: module, class, or function imported
    """

    try:
        importlib.import_module(module_name)
        print(f"'{module_name}' module already installed.")
        return
    except:
        pass

    module_git = "https://github.com/WavefrontEngUB/" + module_name if module_git is None else module_git
    module_path = module_name if module_path is None else str(module_path)

    do_clone = False
    if 'google.colab' in sys.modules:  # You are in google colab
        do_clone = True
        pip_path = module_name
        print("In Google colab detected")
    else:
        if not Path(module_path).exists():
            module_path = input(f"Type the path where you have the {module_name} module "
                                f"(empty to git-clone it in the CWD): ")
            if module_path == "":
                do_clone = True
                pip_path = module_name
            else:
                pip_path = module_path
        else:
            pip_path = module_path
            print(f"Module {module_name} found in {module_path}")

    if do_clone:
        if not Path(pip_path).exists():
            !git clone {module_git} {pip_path}
            if branch is not None:
                !cd {pip_path} && git checkout {branch}

    !pip install -e {'"%s"' % pip_path}