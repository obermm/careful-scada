# careful-scada
Cleaning and preprocessing utilities tailored to the open-source "Wind Turbine SCADA Data For Eary Fault Detection":

Gück, C., & Roelofs, C. (2025). 

*Wind Turbine SCADA Data For Early Fault Detection*. 

Zenodo. 

https://doi.org/10.5281/zenodo.15846963


## Motivation
Cleaning wind turbine SCADA data is time-consuming. Providing reusable and adaptable preprocessing 
utilities lowers the entry burdon for maching learing researches to work with real SCADA data on improving early fault detection.
With open source data and open source code for preprocessing you can start builing models directly.

### Included
- Data Loader: Loading and unifying units + naming conventions of CARE SCADA data
- Preprocessor: Outlier detection and handling
- The Loading and Preprocessing are managed through a central sensor_config.yaml 
- Modelling Framwork: Base Models like naive forecasters are implemented to serve as example.

## Installation (development)
```bash
git clone https://github.com/yourname/WindCAREScale.git
cd WindCAREScale
pip install -e .
```
See the examples/basic_usage.ipynb Notebook and use the citation above to obtain the needed datasets.

