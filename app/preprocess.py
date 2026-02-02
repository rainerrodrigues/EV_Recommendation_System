from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    def __init__(self, numerical_features=None, categorical_features=None, target_col="target_high_efficiency"):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target_col = target_col

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.num_medians = {}
        self.cat_modes = {}

    def fit(self, train_df):
        for col in self.numerical_features:
            self.num_medians[col] = train_df[col].median()

        for col in self.categorical_features:
            self.cat_modes[col] = train_df[col].mode()[0]

        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(train_df[col].fillna(self.cat_modes[col]).astype(str))
            self.label_encoders[col] = le

        X = self._prepare_features(train_df)
        self.scaler.fit(X[self.numerical_features])
        return self

    def transform(self, df):
        X = self._prepare_features(df)
        X[self.numerical_features] = self.scaler.transform(X[self.numerical_features])
        return X, df[self.target_col] if self.target_col in df.columns else None

    def _prepare_features(self, df):
        X = df[self.numerical_features + self.categorical_features].copy()

        for col in self.numerical_features:
            X[col] = X[col].fillna(self.num_medians[col])

        for col in self.categorical_features:
            X[col] = X[col].fillna(self.cat_modes[col]).astype(str)

            mask = ~X[col].isin(self.label_encoders[col].classes_)
            X.loc[mask, col] = self.label_encoders[col].classes_[0]
            X[col] = self.label_encoders[col].transform(X[col])

        return X
