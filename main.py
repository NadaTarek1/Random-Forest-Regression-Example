# Import libraries we will need
import sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# I decided to import joblib directly to avoid a deprecation warning
# that came from the joblib library inside the sklearn library
import joblib

# Save the URL for the CSV of the dataset we will work on from a local directory
dataset_url = 'winequality-red.csv'

# Load the data (Which is separated by the semi-colons) from the URL
data = pd.read_csv(dataset_url, sep=';')

# Testing out some descriptive functions on the data set, no processing done yet
print(data.head())
print(data.shape)
print(data.describe())

# The target feature is the quality, and it will be on the Y axis
y = data.quality
# The other feature (input) will be on the Y axis
X = data.drop('quality', axis=1)

# Produce both taring and test sets for the target and input features, using a test size of 20%
# and a seed of 123, and we stratify the data using target variable, to make training set similiar
# to test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

# Create a model pipeline that scales the data using a standard scaler to standardize the data
# first, then fits a model using a random forest regressor.
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# Print all the hyper possible parameters. Hyper parameters are high level parameters that affect the model
# itself and are not obtained from the data, they are usually a design choice.
# For example, the depth of a decision tree hyper-parameter. i.e a design
# choice taken before training. Not obtained from the data.
# print(pipeline.get_params())


# Store the hyperparameters we want to set for our model in a variable, for more info on the
# hyperparameters of random forest regressor, please visit this link:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# Notice that we store multiple values for each key (for each hyperparameter), how will the model
# know which value to use? This is done by using a method called cross-validation, essentially,
# the code below evaluates the performance of each parameter combination, and chooses
# the best combination.
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit and tune model
clf.fit(X_train, y_train)

# Print the best possible parameters
print(clf.best_params_)

# Use the final model, with the optimum hyperparameters to predict the test dataset
y_pred = clf.predict(X_test)

# Print the evaluation scores using different metrics
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# Finally you can save your final model
joblib.dump(clf, 'rf_regressor.pkl')

# And you can load it too for later use
# clf2 = joblib.load('rf_regressor.pkl')