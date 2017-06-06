#!/usr/bin/env python

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore", category=DeprecationWarning)

seed = 444
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("forestfires.csv", index_col=None)

features = ['temp','RH','wind','rain']

Y = df['area'].values
transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
transformer.fit(Y)
unscaled_Y = Y
Y = transformer.transform(Y)[0]
X = df[features]

input_size = len(features)

def nn1_model():
	model = Sequential()
	model.add(Dense(units=8, input_dim=input_size, activation='relu'))
	model.add(Dense(units=4, activation='relu'))
	model.add(Dense(units=1))
	model.compile(loss='mean_absolute_error', optimizer='adam')
	return model

def nn2_model():
	model = Sequential()
	model.add(Dense(units=100, input_dim=input_size, activation='relu'))
	model.add(Dense(units=1))
	model.compile(loss='mean_absolute_error', optimizer='adam')
	return model

def transformed_mae(y, y_pred):
	y_inverted = transformer.inverse_transform(y)
	y_pred_inverted = transformer.inverse_transform(y_pred)
	return mean_absolute_error(y_inverted, y_pred_inverted)

transformed_mae_scorer = make_scorer(transformed_mae)

def transformed_mse(y, y_pred):
	y_inverted = transformer.inverse_transform(y)
	y_pred_inverted = transformer.inverse_transform(y_pred)
	return mean_squared_error(y_inverted, y_pred_inverted)

transformed_mse_scorer = make_scorer(transformed_mse)

estimators = []
estimators.append(('standardize', MinMaxScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=nn2_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring=transformed_mae_scorer)
print("")
print("{} MAE".format(results.mean()))
results = cross_val_score(pipeline, X, Y, cv=kfold, scoring=transformed_mse_scorer)
print("")
print("{} RMSE".format(np.sqrt(results.mean())))

"""
nn1
13.0190450982923 MAE

64.79967098406117 RMSE

nn2
12.946845400468987 MAE

64.78482275939393 RMSE
"""
