import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        # Convert Date
        X["Date"] = pd.to_datetime(X["Date"])
        X["Day"] = X["Date"].dt.day
        X["Month"] = X["Date"].dt.month
        X["Year"] = X["Date"].dt.year
        X["DayOfWeek"] = X["Date"].dt.dayofweek
        X["IsWeekend"] = X["DayOfWeek"].isin([5,6]).astype(int)

        # Convert Time → Hour
        X["Hour"] = pd.to_datetime(X["Time"], format='%I:%M %p').dt.hour

        # Drop original Date/Time
        X = X.drop(["Date", "Time"], axis=1)

        return X
