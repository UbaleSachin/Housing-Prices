from flask import render_template, Flask, request
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


# route for home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            LotArea=request.form.get('LotArea'),
            OverallQual=request.form.get('OverallQual'),
            OverallCond=request.form.get('OverallCond'),
            YearBuilt=request.form.get('YearBuilt'),
            YearRemodAdd=request.form.get('YearRemodAdd'),
            TotalBsmtSF=request.form.get('TotalBsmtSF'),
            GrLivArea=request.form.get('GrLivArea'),
            BsmtFullBath=request.form.get('BsmtFullBath'),
            TotRmsAbvGrd=request.form.get('TotRmsAbvGrd'),
            Fireplaces=request.form.get('Fireplaces'),
            GarageArea=request.form.get('GarageArea'),
            OpenPorchSF=request.form.get('OpenPorchSF'),
            Street=request.form.get('Street'),
            LotShape=request.form.get('LotShape'),
            LandSlope=request.form.get('LandSlope'),
            Neighborhood=request.form.get('Neighborhood'),
            Foundation=request.form.get('Foundation'),
            CentralAir=request.form.get('CentralAir'),
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        print('Before Predictions')

        prediction_pipeline = PredictPipeline()
        print('Mid Prediction')
        result = prediction_pipeline.predict(pred_df)
        print('After Prediction: ', result)
        return render_template('home.html', result=result[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
