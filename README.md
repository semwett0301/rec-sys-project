# Structure of the project

- `eda` - the folder with exploration analysis of datasets (the name of the file follows the name of a dataset)
- `rec-sys` - the folder that contains trained models with feature selection & engineering for them (the name of the file follows the name of a dataset)
- `frontend` - testing frontend on React 
- `backend` - API for final systems in the 

# Structure of `rec-sys` subfolders
- Every subfolder's name follows the name of the model used in there
- Every subfolder contains required elements:
  - `analysis.ipynb` - this file contains feature selection, feature engineering and creation of prepared dataset processes 
  - `data/train.json` - train dataset
  - `data/test.json` - test dataset
  - `<algorithm>.py` - the file with algorithm that covered by the class that follows common interface
  - `README.MD` - description of the structure of this module
- This folder also contain `common` folder - this is for the code can be shared across all of the subfolders

# Branch rules
- when you work on the code create a branch with name `<your name> - <random i or description of the feature>` 
- apply cross-review approach after PR

# Commit rules
- `feat: <message>` - when you develop a new feature
- `fix: <message>` - when you fix a bug
- `refactor: <message>` - when you refactor existed code