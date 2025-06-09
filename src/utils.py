# src/utils.py

import os
import sys
import dill
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        print(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate a single regression model and return its R^2 score.
    """
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple regression models using GridSearchCV and return a performance report.
    """
    report = {}
    try:
        for model_name, model in models.items():
            if model_name in params:
                grid_search = GridSearchCV(estimator=model, param_grid=params[model_name], cv=3, n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                score = evaluate_model(best_model, X_train, y_train, X_test, y_test)
            else:
                score = evaluate_model(model, X_train, y_train, X_test, y_test)

            report[model_name] = score
        return report
    except Exception as e:
        raise CustomException(e, sys)
