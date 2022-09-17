from copyreg import pickle
from pyexpat import model
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

import os
import config
import model_dispatcher
import argparse
import joblib
import pickle

def predict(fold):

    # reading the csv file
    df_test = pd.read_csv(config.TEST_FILE)

    y_test = np.log(df_test['Price'])

    df_test = df_test.drop(columns=['Price'])

    X_test = df_test.copy()

    model = joblib.load(os.path.join(config.MODEL_PATH,f"model{fold}.bin"))
    y_test_pred = model.predict(X_test)
    print('Prediction of test data')
    r2_score = metrics.r2_score(y_test,y_test_pred)
    mae = metrics.mean_absolute_error(y_test,y_test_pred)
    print(f"Fold={fold}, R2_score={r2_score}, MAE={mae}")

if __name__ == "__main__":
    predict(0)
