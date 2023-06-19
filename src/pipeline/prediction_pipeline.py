import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print('before Loading')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print('after Loading')
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 LotArea: str,
                 OverallQual: str,
                 OverallCond: str,
                 YearBuilt: str,
                 YearRemodAdd: str,
                 TotalBsmtSF: str,
                 GrLivArea: str,
                 BsmtFullBath: str,
                 TotRmsAbvGrd: str,
                 Fireplaces: str,
                 GarageArea: str,
                 OpenPorchSF: str,
                 Street: str,
                 LotShape: str,
                 LandSlope: str,
                 Neighborhood: str,
                 Foundation: str,
                 CentralAir: str,
                 ):
        self.LotArea =LotArea
        self.OverallQual =OverallQual
        self.OverallCond =OverallCond
        self.YearBuilt =YearBuilt
        self.YearRemodAdd =YearRemodAdd
        self.TotalBsmtSF = TotalBsmtSF
        self.GrLivArea = GrLivArea
        self.BsmtFullBath = BsmtFullBath
        self.TotRmsAbvGrd= TotRmsAbvGrd
        self.Fireplaces = Fireplaces
        self.GarageArea = GarageArea
        self.OpenPorchSF = OpenPorchSF
        self.Street = Street
        self.LotShape = LotShape
        self.LandSlope = LandSlope
        self.Neighborhood = Neighborhood
        self.Foundation = Foundation
        self.CentralAir = CentralAir


    def get_data_as_dataframe(self):
        try:
            custom_data_input = {
                'LotArea': [self.LotArea],
                'OverallQual':[self.OverallQual],
                'OverallCond': [self.OverallCond],
                'YearBuilt': [self.YearBuilt],
                'YearRemodAdd': [self.YearRemodAdd],
                'TotalBsmtSF': [self.TotalBsmtSF],
                'GrLivArea': [self.GrLivArea],
                'BsmtFullBath': [self.BsmtFullBath],
                'TotRmsAbvGrd': [self.TotRmsAbvGrd],
                'Fireplaces': [self.Fireplaces],
                'GarageArea':[self.GarageArea],
                'OpenPorchSF': [self.OpenPorchSF],
                'Street': [self.Street],
                'LotShape': [self.LotShape],
                'LandSlope': [self.LandSlope],
                'Neighborhood': [self.Neighborhood],
                'Foundation': [self.Foundation],
                'CentralAir':[self.CentralAir],
            }

            return pd.DataFrame(custom_data_input)

        except Exception as e:
            CustomException(e, sys)