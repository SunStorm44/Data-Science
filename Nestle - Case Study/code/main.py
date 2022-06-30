import os
from src.engine import SalesForecastingEngine


def main():

    # Path to the folder with model inputs
    path = os.path.join(os.getcwd(), 'input')
    # Names of the input files
    filenames = {'x_train': 'X_train.csv',
                 'x_test': 'X_test.csv',
                 'y_train': 'Y_train.csv',
                 'y_test': 'Y_test.csv',
                 }

    # Create the main model engine
    engine = SalesForecastingEngine(num_trees=350)

    # Load the input datasets
    engine.load_data(path, filenames)

    # Process training datasets
    engine.process_training_data(feature_selection_method='rf_importance')

    # Fit the model
    engine.fit_model()

    # Dump the model to external file
    path = os.getcwd()
    name = 'model.p'
    engine.dump_model(path, name)

    # Make predictions
    engine.process_test_data()
    forecasts = engine.make_predictions()

    # Save the forecasted values
    name = 'Y_test.csv'
    forecasts.to_csv(os.path.join(path, name), index=False)

    print('Forecasts have been outputted successfully! Process completed.')


if __name__ == '__main__':
    main()
