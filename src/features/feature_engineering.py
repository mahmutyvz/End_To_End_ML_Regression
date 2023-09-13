def feature_engineering(data):
    """
    This function works as a data augmentation tool to generate new data sets through feature engineering.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing the dataset.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with new features generated through feature engineering.
    """
    df = data.copy()
    # GarageArea: Size of garage in square feet
    # GarageCars: Size of garage in car capacity
    df["GarageCarSize"] = (df["GarageArea"] / df["GarageCars"]).fillna(0)
    # GrLivArea: Above grade (ground) living area square feet
    # TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    df["LivAreaRoomSize"] = df["GrLivArea"] / df["TotRmsAbvGrd"]
    # TotalBsmtSF: Total square feet of basement area
    # 1stFlrSF: First Floor square feet
    # 2ndFlrSF: Second floor square feet
    df["TotalHouseSquareFeet"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    # BsmtFinSF1: Type 1 finished square feet
    # BsmtFinSF2: Type 2 finished square feet
    df["TotalBasementSquareFeet"] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    # FullBath: Full bathrooms above grade
    # BsmtFullBath: Basement full bathrooms
    df["TotalFullBathSize"] = df["FullBath"] + df["BsmtFullBath"]
    # HalfBath: Half baths above grade
    # BsmtHalfBath: Basement half bathrooms
    df["TotalHalfBathSize"] = df["HalfBath"] + df["BsmtHalfBath"]
    # YrSold: Year Sold
    # YearBuilt: Original construction date
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    # YrSold: Year Sold
    # YearRemodAdd: Remodel date
    df["RemodHouseAge"] = df["HouseAge"] - (df["YrSold"] - df["YearRemodAdd"])
    # YearRemodAdd: Remodel date
    # YearBuilt: Original construction date
    df["IsRemod"] = (df["YearRemodAdd"] - df["YearBuilt"]).apply(lambda x: 1 if x > 0 else 0)
    # YrSold: Year Sold
    # GarageYrBlt: Year garage was built
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    # OverallQual: Overall material and finish quality
    # OverallCond: Overall condition rating
    # The quality of the old and new house may be the same, I have added the age of the house to find the difference.
    df["YearHouseQuality"] = (df["OverallQual"] + df["OverallCond"]) / 2 + (2010-df['YearBuilt'])
    # WoodDeckSF: Wood deck area in square feet
    # OpenPorchSF: Open porch area in square feet
    # EnclosedPorch: Enclosed porch area in square feet
    # 3SsnPorch: Three season porch area in square feet
    # ScreenPorch: Screen porch area in square feet
    df['HouseTotalPorchSquareFeet'] = (df['OpenPorchSF']
                                  + df['3SsnPorch']
                                  + df['EnclosedPorch']
                                  + df['ScreenPorch']
                                  + df['WoodDeckSF']
                                 )
    # PoolArea: Pool area in square feet
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    # 2ndFlrSF: Second floor square feet
    df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    # GarageArea: Size of garage in square feet
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    #TotalBsmtSF: Total square feet of basement area
    df['HasBasement'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    # Fireplaces: Number of fireplaces
    df['HasFirePlace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    df = df.drop([
        "GarageArea", "GarageCars", "GrLivArea", 
        "TotRmsAbvGrd", "TotalBsmtSF", "1stFlrSF", 
        "2ndFlrSF", "FullBath", "BsmtFullBath", "HalfBath", 
        "BsmtHalfBath", "YrSold", "YearBuilt", "YearRemodAdd",
        "GarageYrBlt", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
        "ScreenPorch", "OverallQual", "OverallCond",'Utilities',
        'Street','PoolQC','BsmtFinSF1','BsmtFinSF2','HouseAge','WoodDeckSF','PoolArea','2ndFlrSF','Fireplaces'
    ], axis=1)
    return df