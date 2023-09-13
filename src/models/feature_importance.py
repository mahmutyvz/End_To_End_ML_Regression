import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from src.data.preprocess_data import pipeline_build
from src.models.metrics import regression_calculate_scores

def shap_importance(df,target,missing_num_cols,ordinal_col,cat_col,plot=False):
    """
        Calculate feature importances using SHAP values for an XGBoost model.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe containing the features and target variable.
        target : str
            The name of the target variable column in the dataframe.
        missing_num_cols : list
            A list of columns with missing numerical values.
        ordinal_col : list
            A list of ordinal columns.
        cat_col : list
            A list of categorical columns.
        plot : bool, optional
            If True, a bar plot of feature importances will be shown. Default is False.

        Returns
        -------
        shap_values_df : pandas.DataFrame or None
            If plot=False, returns a dataframe containing feature importances based on SHAP values.
            If plot=True, returns None and displays a bar plot of feature importances.
        """
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),
                                                        df[target],
                                                        test_size=0.2,
                                                        random_state=33)
    X_train, X_test = pipeline_build(X_train, X_test, missing_num_cols, ordinal_col, cat_col)
    xgb_params = {
        'random_state': 33,
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.01,
        'gamma': 0.2,
        'min_child_weight': 4,
        'subsample': 1,
        'colsample_bytree': 1
    }

    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    scores = regression_calculate_scores(y_test, y_pred, X_train)
    print("SHAP")
    print(scores)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    shap_values_mean_abs = np.abs(shap_values).mean(axis=0)
    shap_values_df = pd.DataFrame(shap_values_mean_abs,
                                  columns=['importance'],
                                  index=X_test.columns)

    shap_values_df.sort_values('importance',
                               ascending=False,
                               inplace=True)
    if plot:
        plt.figure(figsize=(14, 100))

        sns.barplot(x='importance',
                    y=shap_values_df.index,
                    data=shap_values_df,
                    palette="magma", orient='h',
                    **{'edgecolor': 'purple', 'linewidth': 3, 'alpha': 0.8})

        plt.xlabel('SHAP Değeri', fontsize=16)
        plt.ylabel('Feature', fontsize=16)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.title('Feature Bazında Ortalama SHAP Değerleri', fontsize=16)
        plt.show()
    else:
        return shap_values_df