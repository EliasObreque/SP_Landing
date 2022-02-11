# Solid Propellant for Landing 🚀
This code is used to analysis the feasibility to use solid propellant on the soft-landing problem. To tackle this,
an arraignment of solid propellant engines is proposed, which have multiple independent engines.

📢📢📢 **Code execution can take several hours and use high resources** 📢📢📢


### Imports and requirements 🔧

The principal library used are listed below.
```
numpy 
matplotlib
pandas
scipy
multiprocessing
```

To install run:
```
pip install -r requirements.txt
```

## For simultaneous simulation (Multi Core: 3 Core):

### Run analysis and optimization
The main file is _run_scenarios_multiCore.py_. This script can be run as follows from a console (Anaconda Prompt is 
recommended).:
```
python run_scenarios_multiCore.py
```

📄
**Note: ** This Script calculates the parameters of the control law of each engine. The optimization of these parameters is made by the Genetic Algorithm and using 30 scenarios of uncertainties.
Then, an evaluation is made for 60 scenarios with the updated uncertainties in each scenario.
(See line 309 on _Scenarios/S1D_AFFINE/S1D_AFFINE.py_)

## For simple simulation (1 Core):
### Run analysis and optimization
The main file is _run_scenarios_singleCore.py_. This script can be run as follows from a console (Anaconda Prompt is 
recommended).:
```
python run_scenarios_singleCore.py
```

The default properties for this Script is executed to REGRESSIVE Propellant-Grain-Cross-Section (PGCS) (See line 26, 27 and 28).
You can change the type of PGCS commenting the line 26, and uncomment line 27 or 28.

📄
**Note: ** This Script calculates the parameters of the control law of each engine. The optimization of these parameters is made by the Genetic Algorithm and using 30 scenarios of uncertainties.
Then, an evaluation is made for 60 scenarios with the updated uncertainties in each scenario.
(See line 309 on _Scenarios/S1D_AFFINE/S1D_AFFINE.py_)

## Performance comparison





