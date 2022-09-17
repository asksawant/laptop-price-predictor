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


def run(fold,model):

    # reading the csv file
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is not equal to fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # log transforming the target column
    y_train = np.log(df_train['Price'])
    y_valid = np.log(df_valid['Price'])

    # droping the Price and kfold columns
    df_train = df_train.drop(columns=['Price','kfold'])
    df_valid = df_valid.drop(columns=['Price','kfold'])

    df_valid = df_valid[df_train.columns]

    X_train = df_train.copy()
    X_valid = df_valid.copy()

    # One-Hot-encoding -- STEP1
    step1 = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,12,13])
    ],remainder='passthrough')

    # Regression model -- STEP2
    step2 = model_dispatcher.models[model]

    # Creating Pipeline
    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])

    # fitting the model
    pipe.fit(X_train,y_train)

    y_pred = pipe.predict(X_valid)

    r2_score = metrics.r2_score(y_valid,y_pred)
    mae = metrics.mean_absolute_error(y_valid,y_pred)
    print(f"Fold={fold}, R2_score={r2_score}, MAE={mae}")

if __name__ == "__main__":

    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    # currently we only need fold
    parser.add_argument(
        "--fold",
        type = int
    )

    parser.add_argument(
        "--model",
        type = str
    )

    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(
        fold=args.fold,
        model=args.model
    )
