<a href="https://github.com/dmaluenda/pyHolo"><img src="https://github.com/dmaluenda/pyHolo/blob/master/logo.png" align="left" width="128"></a>
<br>
<br>
<br>

Program to generate holograms for laser beam engineering.

# Dependencies
Check the requirements.txt file. They will be automatically included 
during the installation.

# To install
Pip install this module using the editable mode, either via clonning (devel mode)
or via GitHub wrapper (production mode)

### Production mode
This option is just to use the module, but you cannot modify the source code.

Install this in a certain eviron (conda or venv, as you wish).
```
pip install -e https://github.com/WavefrontEngUB/pyHolo
```
After the installation has finished, the module is inside your
site-packages directory. So you can use it.


### Devel mode
This option is just if you are intended to modify the code.

Install this in a certain eviron (conda or venv, as you wish).
```
git clone --recurse-submodules https://github.com/WavefrontEngUB/pyHolo [location]
pip install -e [location]
```

All changes in the code at `[location]` will be updated in the installed module.

Please, consider to make a Pull Request of your contributions to this repository.


# Usage
The main entry is by command-line in 
```
$ python -m pyHolo --help
positional arguments:
  {gui-production,beam_simulation,slm-cal,holo-gen,utils}
                        Program main mode. See groups below.

options:
  -h, --help            show this help message and exit
  --verbose, -v         To show/print more info. Use '-vv' to show even more.

beam_simulation -> Beam simulation:
  --gui GUI             Numerical Aperture of the system

slm-cal -> SLM calibration mode:
  --label LABEL         Label used to read and write info [Required]
  --SLM 1/2/all         Set it to 1, 2 or 'all' (1 is default)
  --POLs 45/135/all     Set it to 45, 135 or 'all' (all is default)
  --check_ROIs          Display the stored ROIs to check if they are OK.
  --ROIs_points         5 coordinates corresponding to Y1, Y2, Y3, Xi and Xf. Y4 is calculated using the rest.
  --check_peaks         Display the FFT to check where is the peak.
  --freq_peaks FREQ_PEAKS FREQ_PEAKS
                        The two frequency peaks for 45 and 135 degrees.
  --ignore_amp          To ignore the amplitude calibration.
  --use_whole           To ignore ROIs in the amplitude calibration.
  --only_raw            Only estimates the raw response. For first attempts...
  --use_raw_pkl         To avoid the raw estimation and use a precalculated data in the SLM_calibrations/<label>_raw_response.pkl file.

holo-gen -> Hologram generator mode:
  --beam_type N         bar help [Required]
  --NA NA               Numerical Aperture of the system [Required]
  --EP_edges            4 coordinates: Xi, Xf, Yi and Yf of the EP (in pixels), corresponding to the positions where light starts getting in the MO (use a knife edge holo).
  --rho_max RHO_MAX     Radius of the Entrance Pupil on the SLM plane in pixels. It is redundant with --EP_edges

utils -> Some useful tools:
  -s SCRIPT             Run the SCRIPT file. Add --man to check some help. [Required]
  --man                 To show some help for a certain script. Check some typical parameters below:
  --filename FILENAME   Main filename to work.
  --save [FILENAME]     Save the result in 'FILENAME', if not passed it OVERWRITES the original. If it starts or ends with an underscore, it is understood as a suffix.
  --crop [C ...]        Crops images around Cx,Cy center and with a size of Sx and Sy for a rectangular cropping. If just one size is passed then Sy=Sx. [Number of arguments: 3 or 4]

Any additional argument is passed to the final worker. Use: '--extraArg someValue'.
```



**************************************************************************

# Disclaimer

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; check the License.txt for more details.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

 Check also the [LICENSE](LICENSE.txt) file.
*************************************************************************
