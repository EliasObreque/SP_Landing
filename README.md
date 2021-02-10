# SP_Landing

#### For agile run:

_python SolidPropellantLander.py_


#### For setting run:

_python SolidPropellantLander.py Altitude[m] PropellantGeometry Problem Engines N_case_

**Where**: \
Burn propellant geometry: _string_ 'constant' - 'tubular' - 'bates' - 'star'\
Problem: _string_ "noise" - "bias" - "normal" - "bias-noise"\
Engines: _list_ of engines array: 2,4,6 (minimum value is 2 - write without space)\
N_case: Number of case _int_
