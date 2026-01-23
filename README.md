# careful-scada
Cleaning and preprocessing utilities tailored to the open-source wind energy SCADA dataset CARE:
Gück, C., & Roelofs, C. (2025). 
*Wind Turbine SCADA Data For Early Fault Detection*. 
Zenodo. 
https://doi.org/10.5281/zenodo.15846963


## Motivation
Cleaning wind turbine SCADA data is time-consuming. Providing reusable and adaptable preprocessing 
utilities too lower the entry burdon for maching learing researches to work with real SCADA data.
Withth open source data and open source code for preprocessing you can start builing models directly.

### Included
- Data Loader: Loading and unifying units + naming conventions of CARE SCADA data
- Preprocessor: Outlier detection and handling

### Not included
- Model training
- Turbine-specific control logic
- datasets

## Installation (development)
```bash
git clone https://github.com/yourname/WindCAREScale.git
cd WindCAREScale
pip install -e .
```
See the examples/basic_usage.ipynb Notebook. 

## Project structure

care_scada/
├── data_loader.py
├── preprocessing.py
├── thresholds.py
└── rename_map.json

