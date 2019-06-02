import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation

dataset = pd.read_csv("FuelConsumptionCo2.csv")
dataset.head()

df = dataset[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
df.head()

def graph(x, y, x_label, y_label):
    plt.scatter(x, y, color="blue")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


graph(df.ENGINESIZE, df.CO2EMISSIONS, "Engine Size", "CO2 Emissions")
graph(df.CYLINDERS, df.CO2EMISSIONS, "Cylinders", "CO2 Emissions")
graph(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS, "Fuel Consumption", "CO2 Emissions")


n = np.random.rand(len(df)) < 0.8
train = df[n]
test = df[~n]

x_train, y_train = train[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]], \
                   train[["CO2EMISSIONS"]]
x_test, y_test = test[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]], \
                 test[["CO2EMISSIONS"]]


model = Sequential()
model.add(Dense(5, input_dim=3, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss="mse", optimizer="rmsprop", metrics=['mean_absolute_error', 'mean_squared_error'])

model.fit(x_train, y_train, epochs=1000)

model.save('carbon_prediction.h5')

score = model.evaluate(x_test, y_test)
accuracy = score[1]
print(accuracy)
