import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the dataset
df = pd.read_csv('airline_passenger_satisfaction.csv')
# dataset.columns  # Show just names of columns
pd.set_option("display.max_columns", None)
df.head()

df.drop(columns="ID", inplace=True)
df.head()

df.shape

df.describe()

df.dtypes

# Data Cleaning
df.info()  # I lookinf about data in all table

df.isnull().sum()

df["Arrival Delay"]  # Here is some empty sells without data

# I will fill empty cells with mean values
df["Arrival Delay"].mean()

df.isnull().sum()

# Charts
plt.pie(df["Satisfaction"].value_counts(), labels=[
        "Neutral or Dissatisfied", "Satisfied"], autopct="%1.1f%%")
plt.show()

cols = ["Gender", "Customer Type", "Type of Travel", "Class", "Satisfaction"]
plt.figure(figsize=(15, 15))
for i, col in enumerate(cols):
    plt.subplot(3, 2, i + 1)  # 3 rows, 2 columns
    sns.countplot(x=col, data=df)
plt.show()

df.hist(bins=20, figsize=(20, 20), color="orange")
plt.show()

sns.catplot(data=df, x="Age", height=4, aspect=4,
            kind="count", hue="Satisfaction")
plt.show()

sns.catplot(data=df, x="On-board Service", height=4,
            aspect=4, kind="count", hue="Satisfaction")
plt.show()

sns.catplot(data=df, x="Gender", height=4, aspect=4,
            kind="count", hue="Satisfaction")
plt.show()

# Column Data Encoding
df.select_dtypes(include="object").columns

df['Gender'].unique()
df['Customer Type'].unique()
df['Type of Travel'].unique()
df['Class'].unique()
df['Satisfaction'].unique()

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
new_df.head()

new_df.dtypes

# Getting separately the features and the targets
X = new_df.drop(columns='Satisfaction')
X

y = new_df["Satisfaction"].values  # target
y

# Splitting the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Scaling the features
xscaler = MinMaxScaler(feature_range=(0, 1))
X_train = xscaler.fit_transform(X_train)
X_test = xscaler.transform(X_test)

# Scaling the target
yscaler = MinMaxScaler(feature_range=(0, 1))
y_train = yscaler.fit_transform(y_train.reshape(-1, 1))
y_test = yscaler.transform(y_test.reshape(-1, 1))

# Building the Artificial Neural Network
model = Sequential()
model.add(Dense(units=64, kernel_initializer='uniform',
          activation='relu', input_dim=22))
model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# compile describes how I want to train my net
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse',
              metrics=['mean_absolute_error'])

# Training the Artificial Neural Network
model.fit(X_train, y_train, batch_size=10, epochs=100,
          validation_data=(X_test, y_test))

# Making predictions on the test set while reversing the scaling
# y_test = yscaler.inverse_transform(y_test)
y_test = yscaler.inverse_transform(y_test.reshape(-1, 1))  # Reshape y_test
prediction = yscaler.inverse_transform(model.predict(X_test))

# Computing the error rate
loss_error = abs(prediction - y_test)/y_test
print("loss_error: ", np.mean(loss_error))

# Making predictions on the test set while reversing the scaling
y_test = yscaler.inverse_transform(y_test)

y_test = yscaler.inverse_transform(y_test.reshape(-1, 1))  # Reshape y_test
prediction = yscaler.inverse_transform(model.predict(X_test))

# Computing the error rate
loss_error = abs(prediction - y_test)/y_test
print("loss_error: ", np.mean(loss_error))
