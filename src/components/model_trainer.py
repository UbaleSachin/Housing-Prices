import os
import sys

import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_objects
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info('Initializing model trainer')
        try:
            logging.info("Splitting training data and test data")
            # X_train, X_test, y_train, y_test = train_test_split(train_arr, test_arr, test_size=0.2, random_state=42)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
                )

            logging.info("Giving the Dictionary of models")
            models = {
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
            }

            logging.info("Getting the R2 Score and Fitting Train and Test data into Models")

            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                r2 = r2_score(y_test, pred)
                report[list(models.keys())[i]] = r2

            logging.info('Getting best model')
            best_score = max(sorted(report.values()))
            best_model_name = list(report.keys())[list(report.values()).index(best_score)]
            best_model = models[best_model_name]

            if best_score < 0.6:
                raise CustomException("No Best Model Found")

            logging.info('Best found model on both training and testing dataset')
            save_objects(
                file_path=self.model_trainer_config.train_model_file_path,
                ojb=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e, sys)
