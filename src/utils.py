import os
import sys
import pickle

import pandas as pd
import numpy as np

from src.exception import CustomException


def save_objects(file_path, ojb):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(ojb, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

