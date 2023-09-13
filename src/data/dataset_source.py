import pandas as pd
import pickle as pk
from src.models.feature_importance import shap_importance
from sklearn.model_selection import train_test_split
from src.data.preprocess_data import pipeline_build
from src.visualization.visualization import missing_control_plot,missing_count_plot,corr_plot
from paths import Path
def read_dataset(file):
    """
       Read a dataset based on the specified file format.

       Parameters
       ----------
       file : str
           The name of the file to be read (e.g., "data.csv" or "data.xlsx").

       Returns
       -------
       df : pandas.DataFrame
           The read dataset.
   """
    if file.split('.')[1]=='csv':
        df = pd.read_csv(file)
    if file.split('.')[1]=='xlsx':
        df = pd.read_excel(file)
    return df

def final_data_build(file,plot=False):
    """
        If the plot parameter is set to True, the visualization functions will be called and executed.
        The list of selected columns is saved using the pickle module.
        Shap importance function is used to select the best features with the shap feature reduction method.
        The pipeline_build function preprocesses the raw data and prepares it for training.

        Parameters
        ----------
        file : str
            The name of the file to be read (e.g., "data.csv" or "data.xlsx").
        plot : bool, optional
            A flag used to visualize missing values (default is False).

        Returns
        -------
        X_train_shap_columns : pandas.DataFrame
            The processed training dataset after feature extraction.
        y_train : pandas.Series
            The target column of the training dataset.
        X_test_shap_columns : pandas.DataFrame
            The processed test dataset after feature extraction.
        y_test : pandas.Series
            The target column of the test dataset.
    """
    df = read_dataset(file)
    df.drop('Id', axis=1, inplace=True)
    target = 'SalePrice'
    if plot:
        missing_df = missing_control_plot(df)
        missing_count_plot(df,missing_df,variable_type='num')
        corr_plot(df,target)
    col_dict = dict()
    ordinal_col = ['ExterCond', 'HeatingQC', 'ExterQual', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond',
                   'BsmtQual', 'BsmtCond',
                   'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageRangeBuilt', 'OverallQual', 'OverallCond',
                   'KitchenQual', 'PoolQC']
    cat_col = []
    for i in df.columns.tolist():
        if len(df[i].unique()) < 30:
            if i not in ordinal_col:
                cat_col.append(i)
    num_cols = [x for x in df.select_dtypes(include=['int', 'float']).columns.tolist() if
                x not in ordinal_col and x not in cat_col and x != target]
    missing_num_cols = [x for x in df if df[x].isnull().sum() > 0 and x in num_cols]
    col_dict['ordinal_col'] = ordinal_col
    col_dict['cat_col'] = cat_col
    col_dict['num_col'] = num_cols
    col_dict['missing_num_col'] = missing_num_cols
    pk.dump(col_dict, open(Path.colencoder_path, "wb"))
    shap_values_df = shap_importance(df, target, missing_num_cols, ordinal_col, cat_col)
    shap_liste = shap_values_df[:61].index.tolist()
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),
                                                        df[target],
                                                        test_size=0.2,
                                                        random_state=33)
    pk.dump({'shap_columns':shap_liste}, open(Path.shap_cols_path, "wb"))
    X_train, X_test = pipeline_build(X_train, X_test, missing_num_cols, ordinal_col, cat_col)
    X_train_shap_columns = X_train[shap_liste]
    X_test_shap_columns = X_test[shap_liste]
    train = pd.concat([X_train,y_train])
    train.to_csv(Path.cleaned_train_path,index=False)
    return X_train_shap_columns,y_train,X_test_shap_columns,y_test
