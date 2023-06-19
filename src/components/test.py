import numpy as np
import yaml
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd
from src.components import hyperparameter

df = pd.read_csv('E:\\csv\\Housing Prices\\train.csv')

num_columns = df.select_dtypes(exclude='object').columns
cat_columns = df.select_dtypes(include='object').columns

num_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
        ('scaler', StandardScaler(with_mean=False))

    ]
)

preprosessor = ColumnTransformer(
    [
        ("num_pipelne", num_pipeline, num_columns),
        ('cat_pipeline',cat_pipeline, cat_columns)
    ]
)

X = preprosessor.fit_transform(df)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "CatBoost": CatBoostRegressor(),
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "XGBoost": XGBRegressor(),
    }

r2_list = {}
model_list = []


for i in range(len(list(models))):
        model = list(models.values())[i]
        param = hyperparameter.parameters[list(models.keys())[i]]
        gs = GridSearchCV(model, param, cv=3)
        gs.fit(X_train, y_train)
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        r2_list[list(models.keys())[i]] = r2

"""for i in range(len(list(models))):
    model1 = list(models.values())[i]
    param = hyperparameter.parameters[list(models.keys())[i]]
    gs = GridSearchCV(model1, param, cv=3)
    gs.fit(X_train, y_train)
    model1.set_params(**gs.best_params_)
    model1.fit(X_train, y_train)
    after_parameter = model1.predict(X_test)
    r2_after_parameter = r2_score(y_test, after_parameter)
    print(r2_after_parameter)"""

print(r2_list)
best_score = max(sorted(r2_list.values()))
best_model_name = list(r2_list.keys())[list(r2_list.values()).index(best_score)]
best_model = models[best_model_name]
print(best_model)

"""{'CatBoost': 0.9890743877986845, 'Linear Regression': 0.9999999982083568, 'KNN': 0.7763217000384666, 'Decision Tree': 0.9813404662977016, 'Random Forest': 0.991870775200897, 'Gradient Boosting': 0.9954090977233017, 'AdaBoost': 0.9875634811466124, 'XGBoost': 0.9934997792631153}
"""

"""param = hyperparameter.parameters[list(models.keys())[i]]
                gcv = GridSearchCV(model, param, cv=3)
                gcv.fit(X_train, y_train)
                model.set_params(**gcv.best_params_)
                after_parameter = model.fit(X_train, y_train)
                after_parameter_y_test = after_parameter.fit(X_test)
                after_parameter_r2 = r2_score(y_test, after_parameter_y_test)"""

