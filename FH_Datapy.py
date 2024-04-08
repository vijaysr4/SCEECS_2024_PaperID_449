import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("D:/Conference/PhishingDetection/Algo/PASTDF.csv")

print(df.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

df_x = df.iloc[:,1:-1].values # shape (88647, 111)
df_y = df.iloc[:,-1].values # shape (88647,)

print(df_x.shape)



x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)

model = Sequential(
    [               
        tf.keras.Input(shape=(30,)),    #specify input size
        ### START CODE HERE ### 
        #tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(25, activation="sigmoid"),
        tf.keras.layers.Dense(5, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
        ### END CODE HERE ### 
    ], name = "my_model" 
)   

print(model.summary())

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy',
             tf.keras.metrics.Recall(),
             tf.keras.metrics.Precision()]
)

model.fit(
    x_train, y_train,
    epochs=20
)
