import numpy as np
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

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
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        r2_list[list(models.keys())[i]] = r2
print(r2_list)
best_score = max(sorted(r2_list.values()))
best_model_name = list(r2_list.keys())[list(r2_list.values()).index(best_score)]
best_model = models[best_model_name]
print(best_model)
        #r2_list.append(r2)
        #model_list.append(list(models.keys())[i])
"""score = pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model', 'r2score'])
best_model_score = score.loc[score['r2score'] == max(score['r2score'])]
best_model_name = score['Model'].loc[score['r2score'] == max(score['r2score'])]
best_model = models[str(dict(best_model_name).values())]
print(best_model)"""

