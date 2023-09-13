# End_To_End_ML_Regression
### Project Scructure

```
regression/
├─ data/
│  ├─ external/
│  ├─ preprocessed/
│  ├─ raw/
├─ models/
│  ├─ CatBoostRegressor/
│  ├─ LGBMRegressor/
│  ├─ XGBRegressor/
├─ notebooks/
├─ src/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  ├─ visualization/
├─ .gitignore
├─ requirements.txt
├─ README.md
```

### Install the required dependencies.

```shell
pip install -r requirements.txt
```
The root variable in paths.py must be changed to the absolute path of the project.


### Running the Application

You can directly run the application, make train and predictions. 

```bash
streamlit run server.py
```  
