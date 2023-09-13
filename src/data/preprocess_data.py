# -*- coding: utf-8 -*-
import pandas as pd
from src.features.feature_engineering import feature_engineering
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pickle as pk
from paths import Path
def missing_value_fill(data):
    """
     Fill missing values in the dataset with predefined values.

     Parameters
     ----------
     data : pandas.DataFrame
         The input dataframe containing the dataset.

     Returns
     -------
     df : pandas.DataFrame
         The dataframe with missing values filled according to the defined rules.
     """
    df = data.copy()
    # BsmtFinType2: Quality of second finished area (if present)
    # BsmtFinType1: Quality of basement finished area
    # BsmtQual: Height of the basement
    # Fence: Fence quality
    # FireplaceQu: Fireplace quality
    # MasVnrArea: Masonry veneer area in square feet
    # MasVnrType: Masonry veneer type
    # MiscFeature: Miscellaneous feature not covered in other categories
    # PoolQC: Pool quality
    # LotFrontage: Linear feet of street connected to property
    # Alley: Type of alley access
    # BsmtCond: General condition of the basement
    # BsmtExposure: Walkout or garden level basement walls
    # Electrical: Electrical system
    # GarageType: Garage location
    # GarageFinish: Interior finish of the garage
    # GarageQual: Garage quality
    # GarageCond: Garage condition
    # GarageYrBlt: Year garage was built
    # NH : Not have
    # NB : No Basement
    # NF : No Fence
    # NFP : No Fireplace
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('NH')
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('NB')
    df['BsmtQual'] = df['BsmtQual'].fillna('NB')
    df['Fence'] = df['Fence'].fillna('NF')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('NFP')
    # I filled in with 0 as most houses don't have it.
    df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
    # I filled in the empty ones with None since the ones that didn't have the ownership information were labeled as None in the column data.
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MiscFeature'] = df['MiscFeature'].fillna('NH')
    df['PoolQC'] = df['PoolQC'].fillna('NH')
    # I grouped each house by its own neighborhood and filled in the missing values with the median.
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].apply(lambda x : x.fillna(x.median()))
    df['Alley'] = df['Alley'].fillna('NH')
    df['BsmtCond'] = df['BsmtCond'].fillna('NH')
    # I filled in the empty ones with No since the ones that didn't have the ownership information were labeled as No in the column data.
    df['BsmtExposure'] = df['BsmtExposure'].fillna('No')
    # I filled in with the most commonly used system for electrical systems in houses.
    df['Electrical']=df['Electrical'].fillna(df['BsmtExposure'].mode()[0])
    df['GarageType'] = df['GarageType'].fillna('NH')
    df['GarageFinish'] = df['GarageFinish'].fillna('NH')
    df['GarageQual'] = df['GarageQual'].fillna('NH')
    df['GarageCond'] = df['GarageCond'].fillna('NH')
    # To be able to fill in the missing values and group the years,
    #I first filled them with a year that is not in the selected column, then grouped the years,
    #and replaced the year that was not in the column with "NH".
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(2015).astype(int)
    bins = [1900,1909,1919,1929,1939, 1949,1954, 1959,1964, 1969,1974, 1979,1984, 1989,1994,1999,2004,2010,2015]
    labels = ['1900-1909', '1910-1919','1920-1929','1930-1939','1940-1949','1950-1954','1955-1959','1960-1964',
              '1965-1969', '1970-1974', '1975-1979', '1980-1984','1985-1989','1990-1994','1995-1999','2000-2004','2005-2010','NH']
    df['GarageRangeBuilt'] = pd.cut(df['GarageYrBlt'], bins=bins, labels=labels).astype('object')
    return df

def pipeline_build(X_train,X_test,missing_num_cols,ordinal_cols,cat_cols):
    """
    The data is first prepared using the missing_value_fill function, which applies predefined methods to fill in missing value.
    Then, the remaining numeric columns containing NaNs are filled with their medians.
    After ensuring that there are no NaNs left in the data, the feature_engineering function is used to generate new features.
    Categorical columns are transformed using the OrdinalEncoder and OneHotEncoder methods.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training dataset.
    X_test : pandas.DataFrame
        The test dataset.
    missing_num_cols : list
        A list of columns with missing numerical values.
    ordinal_cols : list
        A list of ordinal columns.
    cat_cols : list
        A list of categorical columns.

    Returns
    -------
    X_train : pandas.DataFrame
        The processed training dataset after performing pipeline building steps.
    X_test : pandas.DataFrame
        The processed test dataset after performing pipeline building steps.
    """
    X_train = missing_value_fill(X_train)
    X_test = missing_value_fill(X_test)
    num_dict = dict(zip([], []))
    for missing_col in missing_num_cols:
        if missing_col in X_train.columns.tolist():
            train_median = X_train[missing_col].median()
            dict_temp = train_median
            num_dict[missing_col] = dict_temp
            X_train.loc[:, missing_col] = X_train.loc[:, missing_col].fillna(train_median)
            X_test.loc[:, missing_col] = X_test.loc[:, missing_col].fillna(train_median)
    pk.dump(num_dict, open(Path.missing_num_cols_path, "wb"))
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)
    ordinal_dict = dict(zip([], []))
    ordinal_cols=list(set(ordinal_cols))
    for col in ordinal_cols:
        if col in X_train.columns.tolist():
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            temp_keys = X_train.loc[:, col].values
            temp_values = encoder.fit_transform(X_train.loc[:, col].values.reshape(-1, 1))
            X_train.loc[:, col] = encoder.fit_transform(X_train.loc[:, col].values.reshape(-1, 1))
            X_test.loc[:, col] = encoder.transform(X_test.loc[:, col].values.reshape(-1, 1))
            dict_temp = dict(zip(temp_keys, temp_values))
            ordinal_dict[col] = dict_temp
    pk.dump(ordinal_dict, open(Path.ordinal_cols_path, "wb"))
    for i in cat_cols.copy():
        if i not in X_train.columns.tolist():
            cat_cols.remove(i)

    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train[cat_cols])
    pk.dump(ohe, open(Path.cat_cols_path, "wb"))

    train_ohe = pd.DataFrame(ohe.transform(X_train[cat_cols]).toarray(),
                             columns=ohe.get_feature_names_out(cat_cols))

    val_ohe = pd.DataFrame(ohe.transform(X_test[cat_cols]).toarray(),
                           columns=ohe.get_feature_names_out(cat_cols))

    X_train = pd.concat([X_train.reset_index(drop=True),
                              train_ohe.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True),
                            val_ohe.reset_index(drop=True)], axis=1)

    X_train = X_train.drop(cat_cols,axis=1)
    X_test = X_test.drop(cat_cols,axis=1)
    return X_train,X_test

def test_pipeline_build(test,colpath,missing_num_cols_path,ordinal_cols_path,cat_cols_path,shap_cols_path):
    """
    The function test_pipeline_build provides a similar data preparation method as the pipeline_build function,
    but it utilizes pre-trained data information to prepare the test dataset.
    Parameters
    ----------
    test : pandas.DataFrame
        The test dataset.
    colpath : str
        It contains a list of different column information (pickle file).
    missing_num_cols_path : str
        The file path to load the dictionary of missing numerical columns and their corresponding medians (pickle file).
    ordinal_cols_path : str
        The file path to load the dictionary of ordinal columns and their encoding mappings (pickle file).
    cat_cols_path : str
        The file path to load the one-hot encoder used for transforming categorical columns (pickle file).
    shap_cols_path : str
        The file path to load the list of selected SHAP columns (pickle file).

    Returns
    -------
    test : pandas.DataFrame
        The processed test dataset after performing pipeline building steps.
    """
    test = missing_value_fill(test)
    cols = open(colpath, 'rb')
    missing_num_cols = open(missing_num_cols_path, 'rb')
    ordinal_cols = open(ordinal_cols_path, 'rb')
    cat_cols = open(cat_cols_path, 'rb')
    shap_cols = open(shap_cols_path, 'rb')
    train_cols = pk.load(cols)
    train_missing_num_cols = pk.load(missing_num_cols)
    train_ordinal_cols = pk.load(ordinal_cols)
    train_cat_cols = pk.load(cat_cols)
    train_shap_cols = pk.load(shap_cols)
    cols.close()
    missing_num_cols.close()
    ordinal_cols.close()
    cat_cols.close()
    shap_cols.close()
    for missing_col in train_cols['missing_num_col']:
        if missing_col in test.columns.tolist():
            test.loc[:,missing_col] = test.loc[:, missing_col].fillna(train_missing_num_cols[missing_col])

    test = feature_engineering(test)

    train_cols['ordinal_col'] = list(set(train_cols['ordinal_col']))
    for k1, v1 in train_ordinal_cols.items():
        test.loc[~test[k1].isin(list(v1.keys())), k1] = -1
    for col in train_cols['ordinal_col']:
        if col in test.columns.tolist():
            test.loc[:, col] = test.loc[:, col].replace(train_ordinal_cols[col])

    ohe_cols = train_cols['cat_col']
    for i in ohe_cols.copy():
        if i not in test.columns.tolist():
            ohe_cols.remove(i)
    ohe = train_cat_cols
    ohe_test = pd.DataFrame(ohe.transform(test[ohe_cols]).toarray(),columns=ohe.get_feature_names_out(ohe_cols))

    test = pd.concat([test.reset_index(drop=True),
                         ohe_test.reset_index(drop=True)], axis=1)

    test = test.drop(ohe_cols, axis=1)

    test.drop('Id',axis=1,inplace=True)
    test = test[train_shap_cols['shap_columns']]
    test.to_csv(Path.cleaned_test_path, index=False)
    return test
