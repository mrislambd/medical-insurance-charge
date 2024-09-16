import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize OneHotEncoder for 'smoker' and 'sex'
        self.ohe_smoker_sex = OneHotEncoder(
            drop='first', dtype=int, sparse_output=False)
        self.label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        # Fit the OneHotEncoder on smoker and sex
        self.ohe_smoker_sex.fit(X[['smoker', 'sex']])
        self.label_encoder.fit(X['region'])
        return self

    def transform(self, X):
        X = X.copy()

        # Apply OneHotEncoder to 'smoker' and 'sex'
        smoker_sex_encoded = self.ohe_smoker_sex.transform(
            X[['smoker', 'sex']])
        smoker_sex_columns = ['smoker_yes', 'sex_male']

        # Create DataFrame for encoded variables and merge with original data
        smoker_sex_df = pd.DataFrame(
            smoker_sex_encoded, columns=smoker_sex_columns, index=X.index)
        X = pd.concat([X, smoker_sex_df], axis=1)

        # Label encode the 'region' column
        X['region'] = self.label_encoder.transform(X['region'])

        # Create new features
        X['age_bmi'] = X['age'] * X['bmi']
        X['age_bmi_smoker'] = X['age_bmi'] * X['smoker_yes']

        # Drop original columns
        X = X.drop(columns=['age', 'bmi', 'smoker', 'sex'])

        return X
