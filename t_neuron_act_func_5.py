import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
# Use scikit-learn to grid search the batch size and epochs
from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.models import Sequential
from keras.optimizers import SGD
from keras.models import Sequential
# from tensorflow.keras.layers import Dense
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# from scikeras.wrappers import KerasClassifier

seed = 7
tf.random.set_seed(seed)
# load dataset
df = pd.read_csv('airline_passenger_satisfaction.csv', delimiter=",")
df.drop(columns="ID", inplace=True)
# "inplace=True" mean that I make changes in DataFrame
df["Arrival Delay"].fillna(df["Arrival Delay"].mean(), inplace=True)
df.replace({
    'Gender': {
        'Male': 1,
        'Female': 2,
    },
    'Customer Type': {
        'First-time': 1,
        'Returning': 2,
    },
    'Type of Travel': {
        'Business': 1,
        'Personal': 2,
    },
    'Class': {
        'Business': 1,
        'Economy': 2,
        'Economy Plus': 3,
    },
    'Satisfaction': {
        'Neutral or Dissatisfied': 1,
        'Satisfied': 2,
    },
}, inplace=True)
new_df = df
y = new_df["Satisfaction"].values  # target
X = new_df.drop(columns='Satisfaction')

# Function to create model, required for KerasClassifier


def create_model(activation='relu'):
    # create model
    model = Sequential()
    model.add(Dense(units=22, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model,
                        epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign',
              'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(model__activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
