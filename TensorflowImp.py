import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

df = pd.read_csv("D:/Conference/PhishingDetection/Algo/StandardizedPhishing.csv")


iso_x = pd.read_csv("D:/Conference/PhishingDetection/Algo/isomap_phishing.csv")

df_x = iso_x.iloc[:,:].values # shape (11054, 30)
df_y = df.iloc[:,-1].values # shape (11054,)

print(df_x.shape)
print(df_x[0])

#df_x = df.iloc[:,:-1].values # shape (11054, 30)
'''
from sklearn.manifold import Isomap
# apply isomap with k = 6 and output dimension = 2
model = Isomap(n_components=5, n_neighbors=6)



# Scaling
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
nondim_x_scaled = min_max_scaler.fit_transform(nondim_x_pre)

print(nondim_x_pre.shape)
print(nondim_x_scaled.shape)

nondim_x_scaled = pd.DataFrame(nondim_x_pre, columns=['Column_A', 'Column_B', 'Column_C'])

nondim_x_scaled.to_csv("D:/Conference/PhishingDetection/Algo/LLE_phishing.csv", index=False)

'''
#df_x = df_x.T # shape (30, 11054)

#df_y = df_y.reshape((-1, 1)).T # (1, 11054)

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
print(x_train.shape)

model = Sequential(
    [               
        tf.keras.Input(shape=len(df_x[0,:])),    #specify input size
        ### START CODE HERE ### 
        #tf.keras.layers.Dense(20, activation="relu"),
        Dense(25, activation="relu"),
        Dense(15, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
        ### END CODE HERE ### 
    ], name = "my_model" 
)   

print(model.summary())

model.compile(
    loss= 'binary_crossentropy',
    optimizer = 'Adam',
    metrics=['accuracy', 'Recall', 'Precision']
)

model.fit(
    df_x, df_y,
    epochs=100
)

