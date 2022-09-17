import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":

    # reading the training dataset
    df = pd.read_csv("input/train.csv")

    # features are all columns without price
    # note that there is no id column in the dataset
    # here we have training features
    X = df.drop('Price',axis=1).values

    # Selecting the target
    y = df['Price'].values

    ohe = ColumnTransformer(transformers=[
        ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,12,13])
    ],remainder='passthrough')

    # define the model
    rf = ensemble.RandomForestRegressor()

    pipe = Pipeline([
        ('ohe',ohe),
        ('rf',rf)
    ])

    # define the grid of parameters
    # when using pipeline use prefix of model_name
    # In the pipeline we used the name 'rf' for the estimator step
    # So in the random search hyperparmeter for randomforest
    # should be given with the prefix rf__
    param_grid = {
        "rf__n_estimators":[5,20,50,100],
        # "rf__max_features":['log2','sqrt',1.0],
        "rf__max_depth":[1,2,5,7,11,15],
        # "rf__min_samples_split":[2,6,10],
        # "rf__min_samples_leaf":[1,3,4],
        # "rf__bootstrap":[True, False]
    }

    # initialize grid search
    # when using pipeline
    # Normally, you'd run grid search on the pipeline, not the pipeline on grid search.


    model = model_selection.GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        verbose = 10,
        n_jobs=1,
        cv = 5
    )

    # fit the model and extract best score
    model.fit(X,y)
    print(f"Best score: {model.best_score_}")

    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")