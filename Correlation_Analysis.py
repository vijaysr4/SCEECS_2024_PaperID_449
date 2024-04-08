import pandas as pd
import numpy as np

df = pd.read_csv("D:/Conference/PhishingDetection/Algo/Phishing.csv")
df = df.drop(['Index'], axis=1)
df = df.replace(-1, 0)

import matplotlib.pyplot as plt

f = plt.figure(figsize=(200, 200))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
plt.title('Correlation Matrix', fontsize=10);
plt.show()