#!/usr/bin/env python

import numpy as np 
import pandas as pd 

#Read data
df_train = pd.read_csv("./ecoli.csv")

features = list(df_train.columns.values)
# Remove unwanted features
features.remove('seq_name')
# Remove label
features.remove('class')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode labels to numerical values
le = LabelEncoder()
le.fit(df_train['class'])
df_train['class'] = le.transform(df_train['class'])
class_names = list(le.classes_)

y = df_train['class'].values
x = df_train[features]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=128)
# Check if both chunks have all class labels in them
assert len(set(y_train)) == len(class_names)
assert len(set(y_test)) == len(class_names)

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import np_utils, plot_model

# Simple 3-layer NN
# Linear, tanh activations work about the same.
# Relu, elu are slightly worse
model = Sequential()
model.add(Dense(units=64, activation="linear", input_dim=len(features)))
model.add(Dense(units=32, activation="linear"))
model.add(Dense(units=len(class_names), activation="softmax"))

# Save model visualization to file
plot_model(model, to_file='./model.png', show_shapes=True)

# Adamax optimizer works best
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

#labels should be categorical with softmax
y_train = np_utils.to_categorical(y_train)
gold_labels = y_test
y_test = np_utils.to_categorical(y_test)
a = model.fit(X_train.values, y_train, validation_data=(X_test.values, y_test), callbacks=[early_stopping], epochs=500)

classes = model.predict_classes(X_test.values)

from sklearn import metrics
report = metrics.classification_report(gold_labels, classes, target_names=class_names)
print(report)
"""
val_acc: 0.8929
             precision    recall  f1-score   support

         cp       0.97      0.95      0.96        41
         im       0.76      0.93      0.84        14
        imL       0.00      0.00      0.00         1
        imS       0.00      0.00      0.00         1
        imU       0.83      0.62      0.71         8
         om       1.00      1.00      1.00         5
        omL       1.00      1.00      1.00         3
         pp       0.77      0.91      0.83        11

avg / total       0.88      0.89      0.88        84

"""
