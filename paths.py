class Path:
    """
    The Path class contains various class variables representing file paths, hyperparameter settings,
    and other configurations for a house price prediction project.
    target (str): The target column name for the house price prediction.
    root (str): The root directory path for the project.
    train_path (str): The file path for the raw training dataset.
    cleaned_train_path (str): The file path for the preprocessed and cleaned training dataset.
    test_path (str): The file path for the raw test dataset.
    cleaned_test_path (str): The file path for the preprocessed and cleaned test dataset.
    colencoder_path (str): The file path for the column encoder used for preprocessing categorical columns.
    missing_num_cols_path (str): The file path for the numerical encoder used for handling missing numerical columns.
    ordinal_cols_path (str): The file path for the ordinal encoder used for preprocessing ordinal columns.
    cat_cols_path (str): The file path for the one-hot encoder used for preprocessing categorical columns.
    shap_cols_path (str): The file path for the list of columns used for SHAP (SHapley Additive exPlanations) calculations.
    models_path (str): The directory path where models are saved.
    fold_number (int): The number of folds used in cross-validation during hyperparameter tuning.
    hyperparameter_trial_number (int): The number of hyperparameter trials to be performed during hyperparameter tuning.
    """
    target = 'SalePrice'
    root = 'C:/Users/MahmutYAVUZ/Desktop/Software/Python/kaggle/regression'
    train_path = root+'/data/raw/train.csv'
    cleaned_train_path = root+'/data/preprocessed/cleaned_train.csv'
    test_path = root+'/data/external/test.csv'
    cleaned_test_path = root+'/data/preprocessed/cleaned_test.csv'
    colencoder_path = root+"/data/preprocessed/colencoder.pkl"
    missing_num_cols_path = root+"/data/preprocessed/numencoder.pkl"
    ordinal_cols_path = root+"/data/preprocessed/ordinalencoder.pkl"
    cat_cols_path = root+"/data/preprocessed/oheencoder.pkl"
    shap_cols_path = root+"/data/preprocessed/shapcolumns.pkl"
    models_path = root+"/models/"
    fold_number = 5
    hyperparameter_trial_number = 1