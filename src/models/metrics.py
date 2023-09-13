from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score,mean_absolute_error
from numpy import sqrt
def regression_calculate_scores(y_test,y_pred,X_train):
    """
        Calculate various regression performance scores for the model's predictions.

        Parameters
        ----------
        y_test : pandas.Series
            The true target values of the test set.
        y_pred : array-like
            The predicted target values from the model.
        X_train : pandas.DataFrame
            The training data used for the model.

        Returns
        -------
        scores : dict
            A dictionary containing various regression performance scores.
        """
    scores = {'RMSE' : sqrt(mean_squared_error(y_test, y_pred)),
              'MAE' : mean_absolute_error(y_test, y_pred),
              'RMSLE' : mean_squared_log_error(y_test, y_pred),
              'R2' : r2_score(y_test, y_pred),
              'Adj R2' : 1 - (1 - (r2_score(y_test, y_pred))) * (X_train.shape[0] - 1) / (X_train.shape[0] - len(X_train.columns.tolist()) - 1)}
    return scores