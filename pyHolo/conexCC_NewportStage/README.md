# Newport Stage conexCC LTA
LabView interface to control Newport Stage conexCC (LTA)

## ConexCC labview interface

blablabla

## conexCC_lib dependency  (Newport software)
It is assumed that labView software of conexCC is found in ``./conexCC_lib`` folder, nested into here (ignored by git).

    $ tree conexCC_lib
    ./conexCC_lib
          ├───CONEX
          │   ├───CONEX-CC
          |   |   ├───CONEX-CC Virtual Front Panel    # some code is here
          |   |   └───CONEX-CC.dll    # most of the main code is here
          |   |    
          |   |    (this things below are not used in this repository)
          |   |
          │   ├───CONEX-General
          │   ├───CONEX-LDS
          │   │   └───CONEX-LDS Commands
          │   ├───CONEX-PSD
          │   └───CONEX_IOD
          ├───Conex-CC-GUI-V2.0.0.3_win10
          │   └───Conex-CC GUI_V2.0.0.3
          ├───CONEX_GUI_USB_Driver
          │   └───CONEX_GUI_USB_Driver
          │       └───Windows 64-bit
          ├───ExemplesLabView
          └───UserManuals

If you do not know how to get this conexCC_lib, please contact me!
