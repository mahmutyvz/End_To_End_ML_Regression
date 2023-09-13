from joblib import load
import os
from src.data.dataset_source import read_dataset
from src.data.preprocess_data import test_pipeline_build
from paths import Path
#LOAD MODEL
def predict(model):
    """
       Make predictions using a pre-trained regression model on external data.

       Parameters
       ----------
       model : str
           The name of the pre-trained model to load.

       Returns
       -------
       external_pred : array-like
           The predicted target values for the external data using the loaded model.
       """
    if not os.path.exists(os.path.join(Path.models_path,model,'best_fold.gz')):
        raise FileNotFoundError(f'Best model for {model} could not be found.')
    test = read_dataset(Path.test_path)
    external_data = test_pipeline_build(test,Path.colencoder_path,Path.missing_num_cols_path,Path.ordinal_cols_path,Path.cat_cols_path,Path.shap_cols_path)
    model = load(os.path.join(Path.models_path,model,'best_fold.gz'))
    external_pred = model.predict(external_data)
    return external_pred