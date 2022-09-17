from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
import lightgbm

models = {
    "lr": LinearRegression(),
    "rf": ensemble.RandomForestRegressor(),
    "lgbm": lightgbm.LGBMRegressor()
}