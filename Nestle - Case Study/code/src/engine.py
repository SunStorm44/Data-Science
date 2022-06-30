import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
import pickle


class SalesForecastingEngine:
    def __init__(self, num_trees: int = 100):
        self.model = RandomForestRegressor(num_trees)
        self.input_data = dict()
        self.processed_data = dict()
        self.performance_metrics = dict()

    def load_data(self, path: str, filenames: dict, sep: str = ';', decimal: str = ','):
        # Import data into the appropriate object
        for k, v in filenames.items():
            self.input_data[k] = pd.read_csv(os.path.join(path, v), sep=sep, decimal=decimal)

        print('Input data was loaded successfully...')
        return self

    def process_training_data(self,
                              feature_selection_method: str,
                              num_features: int = 30,
                              nan_threshold: float = .5,
                              low_quantile: float = .1,
                              up_quantile: float = .9):

        def calculate_feature_importance(x: pd.DataFrame,
                                         y: pd.DataFrame,
                                         method: str,
                                         features_num: int = 30) -> pd.DataFrame:

            y = y.iloc[:, -1]
            # Perform Random Forest Importance to understand which features are
            # the most important ones (the ones with greatest decrease in Gini impurity)
            if method == 'rf_importance':
                # Create the random forest and fit it to start training
                model = RandomForestRegressor()
                model.fit(x, y)

                # Extract the importance information of the resulting features
                importance = model.feature_importances_

                # Create a features dataframe and sort it descending
                feat_df = pd.DataFrame({'Features': x.columns, 'Importance': importance})
                feat_df = feat_df.sort_values('Importance', ascending=False)
                # Filter only the top n features
                feat_df = feat_df.head(features_num)
                return feat_df

            elif method == 'rfe':
                lm = LinearRegression()
                rfe = RFE(lm, n_features_to_select=features_num)
                rfe = rfe.fit(x, y)

                feat_df = x.columns[rfe.support_]
                return feat_df

        def calculate_features_score(x: pd.DataFrame, y: pd.DataFrame, check_features: pd.DataFrame) -> dict:

            scores = dict()
            y = y.iloc[:, -1]
            for num in range(1, len(check_features)+1):
                selected_features = check_features.head(num)['Features'].to_list()

                # Filter the training dataset for the selected features
                X_forward = x[selected_features]

                model = RandomForestRegressor()
                model.fit(X_forward, y)

                predictions = model.predict(X_forward)
                r_2 = metrics.r2_score(y, predictions)
                r_2 = round(r_2, 4)

                scores[str(num)] = r_2
            return scores

        # Join x_train and y_train for the feature pre-processing part
        df = pd.merge(self.input_data['x_train'], self.input_data['y_train'], how='inner', on=['key', 'date'])

        # Deal with the NaN values
        df = self.__process_nans(df, nan_threshold)

        X = df.iloc[:, 2:-1]
        Y = pd.DataFrame(df.iloc[:, -1])

        # Trim the outliers in the input data
        X = self.__trim_outliers(X, low_quantile, up_quantile)
        Y = self.__trim_outliers(Y, low_quantile, up_quantile)

        # Assess the features importance
        features_df = calculate_feature_importance(X, Y, feature_selection_method, num_features)
        features_df.to_csv('E:/Projects/Algorithms/Nestle/test.csv', index=False)

        # Pick the final set of features
        features_score = calculate_features_score(X, Y, features_df)

        # Get the final list of features
        selected_features = features_df.head(int(max(features_score)))['Features'].to_list()

        # Filter the training dataset for the selected features
        X = X[selected_features]

        self.processed_data['x_train'] = X
        self.processed_data['y_train'] = Y.iloc[:, -1]
        self.processed_data['features_df'] = features_df
        self.processed_data['features_score'] = features_score

        print('Training data was processed successfully...')
        return self

    def fit_model(self):
        X = self.processed_data['x_train']
        Y = self.processed_data['y_train']

        self.model.fit(X, Y)

        # Get predictions for the training data
        predictions = self.model.predict(X)

        # Create the main performance metrics
        self.__create_metrics(Y, predictions)

        print('Model has been fitted successfully...')

    def process_test_data(self, nan_threshold: float = .5, low_quantile: float = .1, up_quantile: float = .9):
        x_test = self.input_data['x_test'].iloc[:, 2:-1]

        # Get the final list of features
        selected_features = self.processed_data['features_df'].head(int(max(self.processed_data['features_score'])))['Features'].to_list()

        # Filer out all remaining features from the test dataset
        x_test = x_test[selected_features]

        # Process NaNs
        x_test = self.__process_nans(x_test, nan_threshold)

        # Trim outliers
        x_test = self.__trim_outliers(x_test, low_quantile, up_quantile)

        self.processed_data['x_test'] = x_test

        print('Test data has been processed successfully...')

    def make_predictions(self):
        x_test = self.processed_data['x_test']

        # Use the created model to generate predictions for the processed test dataset
        predictions = self.model.predict(x_test)

        # Save the predictions in the inputted table
        y_test = self.input_data['y_test']
        y_test['y'] = predictions

        # Save it in the processed_data object for further use
        self.processed_data['y_test'] = y_test

        print('Predictions performed successfully...')
        return y_test

    def __create_metrics(self, Y: np.array, predictions: np.array):

        mae = metrics.mean_absolute_error(Y, predictions)
        rmse = np.sqrt(metrics.mean_squared_error(Y, predictions))
        r2 = metrics.r2_score(Y, predictions)

        self.performance_metrics['mae'] = mae
        self.performance_metrics['rmse'] = rmse
        self.performance_metrics['r2'] = r2

        print('Mean Absolute Error (MAE):', mae)
        print('Root Mean Squared Error (RMSE):', rmse)
        print('R^2:', r2)

    @staticmethod
    def __process_nans(data: pd.DataFrame, nan_limit: float) -> pd.DataFrame:
        # Create a dict to store information about the NaN values in the dataset
        x_nans = data.isna().sum().to_dict()

        # Remove features with more NaNs then the passed percentage limit
        remove_cols = [k for k, v in x_nans.items() if v > nan_limit * len(data)]
        data = data.drop(remove_cols, axis=1)

        # Remove all the rows where y is 0 or nan
        if 'y' in data.columns:
            data = data.dropna(subset=['y'])
            data = data[data.y != 0]

        # Fill the rest of NaNs with the median for a given feature
        for col in data.columns:
            if any(data[col].isna()):
                data[col] = data[col].fillna(data[col].median())
        return data

    @staticmethod
    def __trim_outliers(data: pd.DataFrame, down_limit: float, up_limit: float) -> pd.DataFrame:
        # Trim the outliers in the input data
        for col in data:
            floor = data[col].quantile(down_limit)
            cap = data[col].quantile(up_limit)

            data[col] = np.where(data[col] < floor, floor, data[col])
            data[col] = np.where(data[col] > cap, cap, data[col])
        return data

    def dump_model(self, path: str, name: str):
        full_name = os.path.join(path, name)

        with open(full_name, 'wb') as model_file:
            pickle.dump(self.model, model_file)

        print('Model has been outputted successfully')

    def load_model(self, path: str, name: str):
        full_name = os.path.join(path, name)

        self.model = pickle.load(open(full_name, 'rb'))

        print('Model has been inputted successfully')
