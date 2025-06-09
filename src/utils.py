import os
import sys
import numpy as np
import pandas as pd
import src.exception as CustomException
import dill


def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    """
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        print(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple regression models and return their scores.
    """
    model_report = {}
    
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score =  1- np.mean((y_test - y_pred) ** 2) / np.var(y_test)
            model_report[model_name] = score
        except Exception as e:
            raise CustomException(e, sys)
    
    return model_report