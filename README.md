# End_To_End_ML_Regression
### Project Scructure

```
regression/
├─ data/
│  ├─ external/
│  ├─ preprocessed/
│  ├─ raw/
├─ images/
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

The root variable in paths.py must be changed to the absolute path of the project.

### Install the required dependencies.

```shell
pip install -r requirements.txt
```


### Running the Application

You can directly run the application, make training and predictions. 

```bash
streamlit run app.py
```  

![Tool Preview 1](https://github.com/mahmutyvz/End_To_End_ML_Regression/blob/1d55fd19ab28e79dd40149428a9897596062bd7d/images/streamlit_1.PNG)
