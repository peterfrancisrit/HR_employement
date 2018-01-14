# Application: HR employement data
#############################

# Build a prediction tool from linear regression to predict satisfaction level based on average monthly work hours 
# and last evaluation. I then built a classification tool on whether or not employees left their jobs based on 
# last evaluation and average monthly hours. 

# REGRESSION 

# Imports
import matplotlib.pyplot as plt
import numpy as np

# Import data using pandas
import pandas as pd
data = pd.read_csv('/Users/harryritchie/Documents/Personal/Employment/HR_comma_sep.csv',delimiter=';',header=0,engine='python')

# Separate response from predictors
y = data['satisfaction_level']
X = data[['last_evaluation', 'average_montly_hours']]

# Split train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=20)


# Create linear regression object
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
model = linear_model.LinearRegression()


# Train the model using the training sets
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)


# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# CLASSIFICATION 


