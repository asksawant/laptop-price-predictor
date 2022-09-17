import numpy as np
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    # reading training data from csv file
    train = pd.read_csv("input/train.csv")

    # creating kfold column having value -1
    train['kfold'] = -1

    # shuffling the data
    train = train.sample(frac=1).reset_index(drop=True)

    # creating bins for stratifiedkfold
    train.loc[:,'bins'] = pd.cut(train['Price'], bins = 3, labels=False)

    # Using StratifiedKFold
    kf = model_selection.StratifiedKFold(n_splits=4)

    for f, (t_,v_) in enumerate(kf.split(X=train, y=train.bins.values)):
        print(len(t_),len(v_))
        train.loc[v_,'kfold'] = f
    
    train = train.drop("bins", axis=1)

    # save the folds to csv
    train.to_csv("input/train_folds.csv", index=False)


