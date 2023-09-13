import streamlit as st
import datetime
import warnings
import pandas as pd
from paths import Path
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from src.models.trainer import Trainer
from src.visualization.visualization import *
from src.models.predict_model import predict
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
st.set_page_config(page_title="End_To_End_Regression",
                   page_icon="chart_with_upwards_trend", layout="wide")
st.markdown("<h1 style='text-align:center;'>House Price Prediction</h1>", unsafe_allow_html=True)
st.write(datetime.datetime.now(tz=None))
tabs = ["Data Analysis", "Visualization", "Train", "Predict", "About"]
page = st.sidebar.radio("Tabs", tabs)
if page == "Data Analysis":
    raw_df = pd.read_csv(Path.train_path)
    option = st.selectbox(
        'While fetching all columns, Streamlit crashes due to a large amount of data. Therefore, which range of columns would you like to retrieve?',
        ('0-20', '20-40', '40-60', '60-81'))
    if option == '0-20':
        df = raw_df.iloc[:, :20]
        variables = {
            "descriptions": {
                "SalePrice": "The property's sale price in dollars. This is the target variable that you're trying to predict.",
                "Id": "Unique ID number of each row of data",
                "MSSubClass": "The building class",
                "MSZoning": "The general zoning classification",
                "LotFrontage": "Linear feet of street connected to property",
                "LotArea": "Lot size in square feet",
                "Street": "Type of road access",
                "Alley": "Type of alley access",
                "LotShape": "General shape of property",
                "LandContour": "Flatness of the property",
                "Utilities": "Type of utilities available",
                "LotConfig": "Lot configuration",
                "LandSlope": "Slope of property",
                "Neighborhood": "Physical locations within Ames city limits",
                "Condition1": "Proximity to main road or railroad",
                "Condition2": "Proximity to main road or railroad (if a second is present)",
                "BldgType": "Type of dwelling",
                "HouseStyle": "Style of dwelling",
                "OverallQual": "Overall material and finish quality",
                "OverallCond": "Overall condition rating",
                "YearBuilt": "Original construction date",
            }
        }
    elif option == '20-40':
        df = raw_df.iloc[:, 20:40]
        variables = {
            "descriptions": {
                "SalePrice": "The property's sale price in dollars. This is the target variable that you're trying to predict.",
                "YearRemodAdd": "Remodel date",
                "RoofStyle": "Type of roof",
                "RoofMatl": "Roof material",
                "Exterior1st": "Exterior covering on house",
                "Exterior2nd": "Exterior covering on house (if more than one material)",
                "MasVnrType": "Masonry veneer type",
                "MasVnrArea": "Masonry veneer area in square feet",
                "ExterQual": "Exterior material quality",
                "ExterCond": "Present condition of the material on the exterior",
                "Foundation": "Type of foundation",
                "BsmtQual": "Height of the basement",
                "BsmtCond": "General condition of the basement",
                "BsmtExposure": "Walkout or garden level basement walls",
                "BsmtFinType1": "Quality of basement finished area",
                "BsmtFinSF1": "Type 1 finished square feet",
                "BsmtFinType2": "Quality of second finished area (if present)",
                "BsmtFinSF2": "Type 2 finished square feet",
                "BsmtUnfSF": "Unfinished square feet of basement area",
                "TotalBsmtSF": "Total square feet of basement area",
                "Heating": "Type of heating",
            }
        }
    elif option == '40-60':
        df = raw_df.iloc[:, 40:60]
        variables = {
            "descriptions": {
                "SalePrice": "The property's sale price in dollars. This is the target variable that you're trying to predict.",
                "HeatingQC": "Heating quality and condition",
                "CentralAir": "Central air conditioning",
                "Electrical": "Electrical system",
                "1stFlrSF": "First Floor square feet",
                "2ndFlrSF": "Second floor square feet",
                "LowQualFinSF": "Low quality finished square feet(all floors)",
                "GrLivArea": "Above grade (ground) living area square feet",
                "BsmtFullBath": "Basement full bathrooms",
                "BsmtHalfBath": "Basement half bathrooms",
                "FullBath": "Full bathrooms above grade",
                "HalfBath": "Half baths above grade",
                "Bedroom": "Number of bedrooms above basement level",
                "Kitchen": "Number of kitchens",
                "KitchenQual": "Kitchen quality",
                "TotRmsAbvGrd": "Total rooms above grade (doesnot include bathrooms)",
                "Functional": "Home functionality rating",
                "Fireplaces": "Number of fireplaces",
                "FireplaceQu": "Fire place quality",
                "GarageType": "Garage location",
                "GarageYrBlt": "Year garage was built",
            }
        }
    elif option == '60-81':
        df = raw_df.iloc[:, 60:]
        variables = {
            "descriptions": {
                "SalePrice": "The property's sale price in dollars. This is the target variable that you're trying to predict.",
                "GarageFinish": "Interior finish of the garage",
                "GarageCars": "Size of garage in car capacity",
                "GarageArea": "Size of garage in square feet",
                "GarageQual": "Garage quality",
                "GarageCond": "Garage condition",
                "PavedDrive": "Paved driveway",
                "WoodDeckSF": "Wood deck area in square feet",
                "OpenPorchSF": "Open porch area in square feet",
                "EnclosedPorch": "Enclosed porch area in square feet",
                "3SsnPorch": "Three season porch area in square feet",
                "ScreenPorch": "Screen porch area in square feet",
                "PoolArea": "Pool area in square feet",
                "PoolQC": "Pool quality",
                "Fence": "Fence quality",
                "MiscFeature": "Miscellaneous feature not covered in other categories",
                "MiscVal": "$Value of miscellaneous feature",
                "MoSold": "MonthSold",
                "YrSold": "YearSold",
                "SaleType": "Type of sale",
                "SaleCondition": "Condition of sale",
            }
        }
    df[Path.target] = raw_df[Path.target]
    profile = ProfileReport(df, title="House Price Prediction", variables=variables, dataset={
        "description": "With 81 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.",
        "url": "https://www.kaggle.com/c/house-prices-advanced-regression-techniques",
    }, )
    st.title("Data Overview")
    st.write(df)
    st_profile_report(profile)
elif page == "Train":
    option = st.radio(
        'What model would you like to use for training?',
        ('XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'))
    if option == 'XGBRegressor':
        model = XGBRegressor(random_state=33)
    elif option == 'LGBMRegressor':
        model = LGBMRegressor(random_state=33)
    elif option == 'CatBoostRegressor':
        model = CatBoostRegressor(random_seed=33)
    trainer = Trainer(Path.train_path, model, Path.models_path, Path.fold_number, Path.hyperparameter_trial_number)

    with st.spinner("Training is in progress, please wait..."):
        cv_result, best_fold_no = trainer.train()
    with st.spinner("Prediction data is being visualized, please wait..."):
        for i in cv_result:
            if i['fold_no'] == best_fold_no:
                st.write("## Best **Model**")
            st.write("Fold No : ", i['fold_no'],
                     "RMSE : ", i['rmse'],
                     "RMSLE : ", i['rmsle'],
                     "MAE : ", i['mae'],
                     "R2 Score : ", i['r2'],
                     "Adj. R2 Score : ", i['adj_r2'], )
            chart_data = pd.DataFrame({'real': i['real'], 'pred': i['pred']})
            line_chart(chart_data, streamlit=True)
elif page == "Predict":
    option = st.radio(
        'What model would you like to use for making predictions?',
        ('XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'))
    pred = predict(option)
    with st.spinner("Predict is in progress, please wait..."):
        st.line_chart(pd.DataFrame(pred, columns=['Prediction']))

elif page == "Visualization":
    df = pd.read_csv(Path.train_path)
    with st.spinner("Visuals are being generated, please wait..."):
        missing_df = missing_control_plot(df, streamlit=True)
        corr_plot(df, Path.target, streamlit=True)
        missing_count_plot(df, missing_df, variable_type='num', streamlit=True)
        missing_count_plot(df, missing_df, variable_type='cat', streamlit=True)
elif page == "About":
    st.header("Contact Info")
    st.markdown("""**mahmutyvz324@gmail.com**""")
    st.markdown("""**[LinkedIn](https://www.linkedin.com/in/mahmut-yavuz-687742168/)**""")
    st.markdown("""**[Github](https://github.com/mahmutyvz)**""")
    st.markdown("""**[Kaggle](https://www.kaggle.com/mahmutyavuz)**""")
st.set_option('deprecation.showPyplotGlobalUse', False)
