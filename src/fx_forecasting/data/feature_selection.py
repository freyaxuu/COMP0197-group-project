import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class FXFeatureSelector:
    def __init__(self, df, target_col, feature_cols):
        self.df = df.copy()
        self.target = target_col
        self.features = feature_cols
        self.X = self.df[feature_cols]
        self.y = self.df[target_col]

    def remove_highly_correlated(self, threshold=0.9):
        """Removes one of two features that are highly correlated."""
        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        print(f"Dropping {len(to_drop)} redundant features: {to_drop}")
        self.features = [f for f in self.features if f not in to_drop]
        self.X = self.df[self.features]
        return self.features

    def get_mutual_info_scores(self):
        """Estimates mutual information for a continuous target."""
        # MI requires no NaNs
        mi_scores = mutual_info_regression(self.X, self.y, random_state=42)
        mi_series = pd.Series(mi_scores, name="MI Scores", index=self.X.columns)
        return mi_series.sort_values(ascending=False)

    def run_rfecv(self, cv_splits=5):
        """Recursive Feature Elimination with Cross-Validation."""
        # Using a simpler model like RandomForest for selection speed
        estimator = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        
        # Note: In production, use TimeSeriesSplit here!
        from sklearn.model_selection import TimeSeriesSplit
        selector = RFECV(
            estimator=estimator, 
            step=1, 
            cv=TimeSeriesSplit(n_splits=cv_splits),
            scoring='neg_mean_squared_error'
        )
        selector.fit(self.X, self.y)
        
        selected = list(self.X.columns[selector.support_])
        print(f"RFECV selected {len(selected)} features.")
        return selected