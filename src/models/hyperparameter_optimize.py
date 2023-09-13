import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
import copy
def optuna_optimize(X, y, model, fold_object,step):
    """
        Perform hyperparameter optimization using Optuna for a given regression model.

        Parameters
        ----------
        X : pandas.DataFrame
            The input dataframe containing the features.
        y : pandas.Series
            The target variable series.
        model : estimator object
            The regression model to be optimized.
        fold_object : cross-validation object
            The cross-validation object used for training and validation.
        step : int
            The number of optimization steps or trials to perform.

        Returns
        -------
        best_params : dict
            The best hyperparameter combination found through optimization.
        best_value : float
            The best value (score) achieved by the optimized model.
        """
    print("Model : ", type(model).__name__)
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate',0.0005, 0.5,step=0.01),
            'n_estimators': trial.suggest_int('n_estimators',0, 3000,step=10),
            'max_bin': trial.suggest_int('max_bin',16, 2048,step=16),
            'subsample': trial.suggest_float('subsample', 0.1, 1,step=0.1),
        }
        if type(model).__name__ == 'CatBoostRegressor':
            params.update({
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1, step=0.1),
                'loss_function' : 'RMSE',
                'max_depth': trial.suggest_int('max_depth', 6, 10),
                'random_seed': 33,
                'verbose': False
            })
        elif type(model).__name__ == 'XGBRegressor':
            params.update({
                'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.1, 1, step=0.1),
                'max_depth' : trial.suggest_int('max_depth', 1, 16, step=1),
                'verbosity': 0
            })
        elif type(model).__name__ == 'LGBMRegressor':
            params.update({
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1, step=0.1),
                'verbosity': 0,
            })
        liste = []
        for fold_no, (train_index, test_index) in enumerate(fold_object.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model_copy = copy.deepcopy(model)
            model_copy.set_params(**params)
            model_copy.fit(X_train, y_train)
            liste.append(np.sqrt(mean_squared_error(y_test, model_copy.predict(X_test))))
        return np.mean(liste)

    study = optuna.create_study(direction='minimize', study_name='regression')
    study.optimize(objective, n_trials=step)
    print(f"Best Params : {study.best_params}",
          f"Best Value : {study.best_value}")
    return study.best_params, study.best_value
