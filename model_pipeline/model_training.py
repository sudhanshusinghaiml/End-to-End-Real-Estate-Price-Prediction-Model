import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def data_split(df, target_variable, size, seed):
    """
        df: dataframe
        target_variable: target feature name
        size: test size ratio
        seed: random state
    """
    try:
        X = df.drop(target_variable, axis=1)
        y = df[[target_variable]]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)
    except Exception as e:
        print('Error in data_split function', e)
    else:
        return x_train, x_test, y_train, y_test


def regression_model_training(x_train, x_test, y_train, y_test):
    """
        Script to train linear regression and regularization models
        :param x_train: training split
        :param y_train: training target vector
        :param x_test: test split
        :param y_test: test target vector
        :return: DataFrame of model evaluation, model objects
    """
    try:
        models = [
            ('linear_regression', LinearRegression()),
            ('ridge_regulation', Ridge()),
            ('lasso_regulation', Lasso())
        ]
        output_df = pd.DataFrame()
        columns_for_comparison = ['Model', 'Training_R2', 'Training_MAE', 'Training_RMSE', 'Test_R2', 'Test_MAE',
                                  'Test_RMSE']
        model_estimators = []

        for name, model in models:
            # Model training
            regression = model.fit(x_train, y_train)
            y_prediction_training = regression.predict(x_train)
            y_prediction_test = regression.predict(x_test)

            # Evaluating Models - Mean Absolute Error or MAE
            MAE_training = round(mean_absolute_error(y_train, y_prediction_training), 6)
            MAE_test = round(mean_absolute_error(y_test, y_prediction_test), 6)

            # Evaluating Models - Root Mean Squared Error or RMSE
            RMSE_training = round(mean_squared_error(y_train, y_prediction_training, squared=False), 6)
            RMSE_test = round(mean_squared_error(y_test, y_prediction_test, squared=False), 6)

            # Evaluating Models - R2 Score
            R2_training = round(r2_score(y_train, y_prediction_training), 6)
            R2_test = round(r2_score(y_test, y_prediction_test), 6)
            model_estimators.append(regression)

            # comparison dataframe
            metrics_score = [name, R2_training, MAE_training, RMSE_training, R2_test, MAE_test, RMSE_test]
            score_dict = dict(zip(columns_for_comparison, metrics_score))
            df_score = pd.DataFrame([score_dict])
            output_df = pd.concat([output_df, df_score], ignore_index=True)

    except Exception as e:
        print('Error in regression_model_training function', e)
    else:
        return output_df, model_estimators[0], model_estimators[1], model_estimators[2]


def ensemble_model_training(x_train, x_test, y_train, y_test, model_estimators):
    """
        Script to train a voting regressor
        estimators: List of tuples of name and fitted regressor objects
    """
    try:
        columns_for_comparison = ['Model', 'Training_R2', 'Training_MAE', 'Training_RMSE', 'Test_R2', 'Test_MAE',
                                  'Test_RMSE']

        # Model training
        voting_ensemble = VotingRegressor(estimators=model_estimators)
        voting_ensemble.fit(x_train, y_train)
        y_prediction_training = voting_ensemble.predict(x_train)
        y_prediction_test = voting_ensemble.predict(x_test)

        # Evaluating Models - Mean Absolute Error or MAE
        MAE_training = round(mean_absolute_error(y_train, y_prediction_training), 6)
        MAE_test = round(mean_absolute_error(y_test, y_prediction_test), 6)

        # Evaluating Models - Root Mean Squared Error or RMSE
        RMSE_training = round(mean_squared_error(y_train, y_prediction_training, squared=False), 6)
        RMSE_test = round(mean_squared_error(y_test, y_prediction_test, squared=False), 6)

        # Evaluating Models - R2 Score
        R2_training = round(r2_score(y_train, y_prediction_training), 6)
        R2_test = round(r2_score(y_test, y_prediction_test), 6)

        # comparison dataframe
        metrics_score = ['Voting_Ensemble', R2_training, MAE_training, RMSE_training, R2_test, MAE_test, RMSE_test]
        score_dict = dict(zip(columns_for_comparison, metrics_score))
        df_score = pd.DataFrame([score_dict])

    except Exception as e:
        print('Error in ensemble_regressor function', e)
    else:
        return df_score, voting_ensemble
