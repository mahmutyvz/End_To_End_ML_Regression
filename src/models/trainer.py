import pandas as pd
from sklearn.model_selection import KFold
from src.data.dataset_source import final_data_build
from src.models.hyperparameter_optimize import optuna_optimize
import os
from joblib import dump
from src.models.metrics import regression_calculate_scores
class Trainer:
    def __init__(self,train_path,model,saved_model_path,fold_number,hyperparameter_trial):
        """
        Initializes the Trainer class.

        Parameters:
            train_path (str): File path to the training dataset.
            model (object): Regression model object (e.g., XGBRegressor, LGBMRegressor, CatBoostRegressor).
            saved_model_path (str): Directory path where the trained model will be saved.
            fold_number (int): Number of folds to be used in K-Fold cross-validation.
            hyperparameter_trial (int): Number of hyperparameter optimization trials using Optuna.

        Returns:
            None
        """
        self.train_path = train_path
        self.model = model
        self.saved_model_path = saved_model_path
        self.fold_number = fold_number
        self.hyperparameter_trial = hyperparameter_trial
    def train(self):
        """
        Trains the regression model using K-Fold cross-validation and hyperparameter optimization with Optuna.

        Returns:
            cv_results (list): A list of dictionaries containing the evaluation scores of each fold.
                               Each dictionary contains the following keys:
                               - 'fold_no': Fold number.
                               - 'rmse': Root Mean Squared Error.
                               - 'mae': Mean Absolute Error.
                               - 'rmsle': Root Mean Squared Logarithmic Error.
                               - 'r2': R-squared.
                               - 'adj_r2': Adjusted R-squared.
                               - 'real': True target values for the validation set.
                               - 'pred': Predicted target values for the validation set.

            best_fold_no (int): The fold number with the lowest RMSE value, used to save the best model.
        """
        X_train,y_train,X_test,y_test = final_data_build(self.train_path)
        kf = KFold(n_splits=self.fold_number, shuffle=True, random_state=33)
        best_params,best_value = optuna_optimize(X_train,y_train,self.model,kf,self.hyperparameter_trial)
        self.model.set_params(**best_params)
        X = pd.concat([X_train,X_test],axis=0)
        y = pd.concat([y_train,y_test],axis=0)
        cv_results = []
        directory = os.path.join(self.saved_model_path)
        if not os.path.exists(os.path.join(directory,str(f'{type(self.model).__name__}'))):
            os.makedirs(os.path.join(directory, str(f'{type(self.model).__name__}')), exist_ok=True)
        for fold_no, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            self.model.fit(X_train, y_train,
                      eval_set=[(X_train,y_train),(X_test,y_test)],
                      early_stopping_rounds=200,verbose=0)
            y_pred = self.model.predict(X_test)
            scores = regression_calculate_scores(y_test, y_pred, X_train)
            print(f'{str(type(self.model).__name__)} Fold No : ', fold_no)
            print(f'Scores {scores}')
            cv_results.append({'fold_no': fold_no,
                               'rmse': scores['RMSE'],
                               'mae': scores['MAE'],
                               'rmsle': scores['RMSLE'],
                               'r2': scores['R2'],
                               'adj_r2': scores['Adj R2'],
                               'real': y_test.tolist(), 'pred': y_pred.tolist()})
            model_name = os.path.join(f'{directory}/{str(type(self.model).__name__)}/{str(fold_no)}.gz')
            dump(self.model, model_name, compress=('gzip', 3))
        min_rmse_dict = min(cv_results, key=lambda x: x['rmse'])
        best_fold_no = min_rmse_dict['fold_no']
        if os.path.exists(os.path.join(f'{directory}/{str(type(self.model).__name__)}/best_fold.gz')):
            os.remove(os.path.join(f'{directory}/{str(type(self.model).__name__)}/best_fold.gz'))
        os.rename(os.path.join(f'{directory}/{str(type(self.model).__name__)}/{str(best_fold_no)}.gz'),os.path.join(f'{directory}/{str(type(self.model).__name__)}/best_fold.gz'))
        print("Best fold",best_fold_no,best_value,best_params)
        return cv_results,best_fold_no