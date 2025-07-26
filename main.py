import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error


train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

# Cleaning of train dataset 

high_null_cols = (train_data.isnull().mean() > 0.8)
cols_to_drop = high_null_cols[high_null_cols].index
train_data.drop(cols_to_drop, axis=1, inplace=True)

for col in train_data.columns:
    if train_data[col].dtype == 'object':  # categorical
        train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    else:  # numeric
        train_data[col] = train_data[col].fillna(train_data[col].mean())

cat_cols = train_data.select_dtypes(include='object').columns
new_cat_cols=pd.get_dummies(train_data[cat_cols],drop_first=True)
train_data.drop(cat_cols,axis=1,inplace=True)
train_data = pd.concat([train_data, new_cat_cols], axis=1)


# Cleaning of test dataset 

high_null_cols_1=(test_data.isnull().mean()>0.8)
cols_to_drop_1=high_null_cols_1[high_null_cols_1].index
test_data.drop(cols_to_drop_1,axis=1,inplace=True)

for cols in test_data.columns:
    if test_data[cols].dtype=='object':
        test_data[cols]=test_data[cols].fillna(test_data[cols].mode()[0])
    else:
        test_data[cols]=test_data[cols].fillna(test_data[cols].mean())

cat_cols_1=test_data.select_dtypes(include='object').columns
new_cat_cols_1=pd.get_dummies(test_data[cat_cols_1],drop_first=True)
test_data.drop(cat_cols_1,axis=1,inplace=True)
test_data=pd.concat([test_data,new_cat_cols_1],axis=1)

test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

for cols in train_data.columns:
    if cols in test_data.columns:
        test_data[cols]=test_data[cols].astype(train_data[cols].dtype)

# Scaling
scaler=StandardScaler()

train_data_scaled=scaler.fit_transform(train_data)
test_data_scaled=scaler.fit_transform(test_data)

train_data_scaled_df=pd.DataFrame(train_data_scaled,columns=train_data.columns)
test_data_scaled_df=pd.DataFrame(test_data_scaled,columns=test_data.columns)

X=train_data_scaled_df.drop(['SalePrice','Id'],axis=1)
Y=train_data['SalePrice']



#### Feature selection with coorelation method 
# This is the performance
# R² Score: 0.6898455416778145
# MSE: 1570711805.6632442
# RMSE: 39632.206671635686

# corrs=train_data.corr()['SalePrice'].abs().sort_values(ascending=False)
# high_corrs=corrs[1:11].index

# X_corr=X[high_corrs]

# x_train,x_test,y_train,y_test=train_test_split(X_corr,Y,test_size=0.2, random_state=42)

# lr=LinearRegression()
# lr.fit(x_train,y_train)

# y_predict=lr.predict(x_test)

# r2=r2_score(y_predict,y_test)
# mse=mean_squared_error(y_predict,y_test)
# rmse=mse**(0.5)

# print("R² Score:", r2)
# print("MSE:", mse)
# print("RMSE:", rmse)



####Feature selection with help of Wrapper method
# R² Score: 0.8443194815202413
# MSE: 1003403269.2946824
# RMSE: 31676.5413089037

# model=Lasso(alpha=0.1,max_iter=100000)
# rfe=RFE(model,n_features_to_select=20)
# rfe.fit(X,Y)

# selected_features=X.columns[rfe.support_]
# X_rfe=X[selected_features]

# x_train,x_test,y_train,y_test=train_test_split(X_rfe,Y,test_size=0.2, random_state=42)

# ridge=Ridge(alpha=0.1)
# ridge.fit(X_rfe,Y)

# y_predict=ridge.predict(x_test)

# r2=r2_score(y_predict,y_test)
# mse=mean_squared_error(y_predict,y_test)
# rmse=mse**(0.5)

# print("R² Score:", r2)
# print("MSE:", mse)
# print("RMSE:", rmse)











