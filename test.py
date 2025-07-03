import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

# high_null_cols = (train_data.isnull().mean() > 0.8)
# cols_to_drop = high_null_cols[high_null_cols].index
# train_data.drop(cols_to_drop, axis=1, inplace=True)

# # print(train_data.isnull())

# # train_data=train_data.fillna(0)
# train_data=train_data.drop(train_data.isnull().mean()>0.8)
# plt.figure(figsize=(26, 10))
# sns.heatmap((train_data.isnull().mean()>0.8).to_frame().T)

# plt.show()
train_data.info()
test_data.info()