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