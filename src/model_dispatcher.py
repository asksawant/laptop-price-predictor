from sklearn import tree
from sklearn import ensemble
from sklearn.linear_model import LinearRegression

models = {
    "lr": LinearRegression(),
    "rf": ensemble.RandomForestRegressor()
}