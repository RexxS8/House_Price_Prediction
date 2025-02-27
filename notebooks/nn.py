# -*- coding: utf-8 -*-
"""NN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DXu9Zh-PZa2bW0mU66yW0i6i4rPKZsH1
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/andri/Documents/Andri Martin/Coding/Prediksi/data/Realestate.csv')
df.head()

df.shape

df.info()

df['X1 transaction date'] = df['X1 transaction date'].astype(int)

df.head()

df.columns

new_names = ["No", "transaction_date", "house_age", "dist_to_MRT", "conv_store", "lat", "long", "house_price"]
df = df.set_axis(new_names, axis="columns")

df.head()

df.set_index("No")

print(df.nunique())

dups = df.duplicated()
print(dups.any())

df.describe()

df.boxplot(column=["transaction_date"])

df.boxplot(column=["house_age"])

df.boxplot(column=["dist_to_MRT"])

df.boxplot(column=["conv_store"])

df.boxplot(column=["lat"])

df.boxplot(column=["long"])

df.boxplot(column=["house_price"])

Q1 = df.dist_to_MRT.quantile(0.25)
Q3 = df.dist_to_MRT.quantile(0.75)
IQR = Q3 - Q1
df_1 = df[~((df.dist_to_MRT < (Q1 - 1.5 * IQR)) | (df.dist_to_MRT > (Q3 + 1.5 * IQR)))] # df['dist_to_MRT']
print(df.shape)
print(df_1.shape)

Q1 = df_1.lat.quantile(0.25)
Q3 = df_1.lat.quantile(0.75)
IQR = Q3 - Q1
df_2 = df_1[~((df_1.lat < (Q1 - 1.5 * IQR)) | (df_1.lat > (Q3 + 1.5 * IQR)))]
print(df_1.shape)
print(df_2.shape)

Q1 = df_2.long.quantile(0.25)
Q3 = df_2.long.quantile(0.75)
IQR = Q3 - Q1
df_3 = df_2[~((df_2.long < (Q1 - 1.5 * IQR)) | (df_2.long > (Q3 + 1.5 * IQR)))]
print(df_2.shape)
print(df_3.shape)

Q1 = df_3.house_price.quantile(0.25)
Q3 = df_3.house_price.quantile(0.75)
IQR = Q3 - Q1
df_4 = df_3[~((df_3.house_price < (Q1 - 1.5 * IQR)) | (df_3.house_price > (Q3 + 1.5 * IQR)))]
print(df_3.shape)
print(df_4.shape)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_5 = df [~((df < (Q1 - 1.5 * IQR )) | (df > (Q3 + 1.5* IQR))).any(axis=1)]
print(df.shape)
print(df_5.shape)

df_5.info()

import matplotlib.pyplot as plt

# Membuat subplot dengan 1 baris dan 3 kolom
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Boxplot untuk df
df.boxplot(column=["dist_to_MRT"], ax=axes[0])
axes[0].set_title("Boxplot df")

# Boxplot untuk df_4
df_4.boxplot(column=["dist_to_MRT"], ax=axes[1])
axes[1].set_title("Boxplot df_4")

# Boxplot untuk df_5
df_5.boxplot(column=["dist_to_MRT"], ax=axes[2])
axes[2].set_title("Boxplot df_5")

# Mengatur judul keseluruhan
plt.suptitle("Boxplot of dist_to_MRT")

# Menampilkan plot
plt.show()

df.columns

# menggunakan scatterplots untuk cek hubungan linier antara house price dengan kolom lainnya
p = sns.pairplot(df, x_vars=['transaction_date', 'house_age', 'dist_to_MRT', 'conv_store',
       'lat', 'long'], y_vars= 'house_price', size=4, aspect=0.7)

#cek data setelah dikeluarkan outliernya
p = sns.pairplot(df_4, x_vars=['transaction_date', 'house_age', 'dist_to_MRT', 'conv_store',
       'lat', 'long'], y_vars= 'house_price', size=4, aspect=0.7)

p = sns.pairplot(df_5, x_vars=['transaction_date', 'house_age', 'dist_to_MRT', 'conv_store',
       'lat', 'long'], y_vars= 'house_price', size=4, aspect=0.7)

df_model_house_age_l_20 = df[df['house_age'] < 20]
df_model_house_age_m_20 = df[df['house_age']> 20]

p = sns.pairplot(df_model_house_age_l_20, x_vars=['transaction_date', 'house_age', 'dist_to_MRT', 'conv_store',
       'lat', 'long'], y_vars= 'house_price', size=4, aspect=0.7)

p = sns.pairplot(df_model_house_age_m_20, x_vars=['transaction_date', 'house_age', 'dist_to_MRT', 'conv_store',
       'lat', 'long'], y_vars= 'house_price', size=4, aspect=0.7)

sns.heatmap(df.corr(), annot= True)

sns.heatmap(df_4.corr(), annot= True)

sns.heatmap(df_5.corr(), annot= True)

sns.heatmap(df_model_house_age_l_20.corr(), annot= True)

sns.heatmap(df_model_house_age_m_20.corr(), annot= True)

from sklearn.model_selection import train_test_split
X = df.drop(['house_price'],axis=1) # data fitur/prediktor
y = df.house_price # target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit (X_train, y_train)

reg.coef_

y_pred = reg.predict(X_test)
y_pred

from sklearn.metrics import r2_score
print('r2 Score : ', r2_score(y_test, y_pred))

df_4.columns

df_4['trans_dist_to_MRT'] =  np.log(df_4.dist_to_MRT.copy())
p = sns.pairplot(df_4, x_vars=['trans_dist_to_MRT'], y_vars= 'house_price', size=4, aspect=0.7)

df_5['trans_dist_to_MRT'] =  np.log(df_5.dist_to_MRT.copy())
p = sns.pairplot(df_5, x_vars=['trans_dist_to_MRT'], y_vars= 'house_price', size=4, aspect=0.7)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#X_1 = df_4.drop(['house_price','transaction_date', 'house_age', 'conv_store','lat', 'trans_dist_to_mrt'],axis=1)
#X_1 = df_4.drop(['house_price','transaction_date','long'],axis=1)
#X_1 = df_4.drop(['No', 'transaction_date', 'house_age', 'conv_store','lat', 'long','trans_dist_to_mrt', 'house_price'],axis=1)

X_1 = df_5[['house_age', 'dist_to_MRT', 'conv_store']].copy()
X_2 = df_5[['house_age', 'trans_dist_to_MRT', 'conv_store']].copy()
X_3 = df[['house_age', 'dist_to_MRT', 'conv_store']].copy()
y = df_5['house_price']
y1 = df['house_price']

X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y, test_size=0.05, random_state=0)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y, test_size=0.05, random_state=0)
X_3_train, X_3_test, y_3_train, y_3_test = train_test_split(X_3, y1, test_size=0.05, random_state=0)

reg_1 = LinearRegression()
reg_1.fit(X_1_train, y_1_train)
reg_2 = LinearRegression()
reg_2.fit(X_2_train, y_2_train)
reg_3 = LinearRegression()
reg_3.fit(X_3_train, y_3_train)

print('Data bersih')
print(f'reg_1.coef_: {reg_1.coef_}')
y_1_pred = reg_1.predict(X_1_test)
print('R-squared Score for model reg_1: ', r2_score(y_1_test, y_1_pred))
print('MAE',mean_absolute_error(y_1_test, y_1_pred))
print('MSE',mean_squared_error(y_1_test, y_1_pred))

print('\nData bersih yang di logaritma')
print(f'reg_2.coef_: {reg_2.coef_}')
y_2_pred = reg_2.predict(X_2_test)
print('R-squared Score for model reg_2: ', r2_score(y_2_test, y_2_pred))

print('\nDataset awal')
print(f'reg_3.coef_: {reg_3.coef_}')
y_3_pred = reg_3.predict(X_3_test)
print('R-squared Score for model reg_2: ', r2_score(y_3_test, y_3_pred))

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Membaca data DataFrame df_4 seperti yang Anda lakukan sebelumnya

# Memisahkan fitur dan target
X_1 = df_5[['house_age', 'dist_to_MRT', 'conv_store']].copy()
X_2 = df_5[['house_age', 'trans_dist_to_MRT', 'conv_store']].copy()
y = df_5['house_price']

# Normalisasi fitur
scaler = StandardScaler()
X_1 = scaler.fit_transform(X_1)
X_2 = scaler.fit_transform(X_2)

# Memisahkan data menjadi data latih dan data uji
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1, y, test_size=0.05, random_state=0)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y, test_size=0.05, random_state=0)

# Parameter untuk MLPRegressor
params = {
    'hidden_layer_sizes': (15, 10),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0,
    'batch_size': 10,
    'random_state': 0,
    'tol': 0.0001,
    'nesterovs_momentum': False,
    'learning_rate': 'constant',
    'learning_rate_init': 0.01,
    'max_iter': 500,
    'shuffle': True,
    'n_iter_no_change': 50,
    'verbose': False
}

# Membuat model MLPRegressor
regressor_1 = MLPRegressor(**params)
regressor_2 = MLPRegressor(**params)

# Melatih model
regressor_1.fit(X_1_train, y_1_train)
regressor_2.fit(X_2_train, y_2_train)

# Membuat prediksi
y_1_pred = regressor_1.predict(X_1_test)
y_2_pred = regressor_2.predict(X_2_test)

# Menghitung metrik evaluasi untuk y1
mae1 = mean_absolute_error(y_1_test, y_1_pred)
mse1 = mean_squared_error(y_1_test, y_1_pred)
r_squared1 = r2_score(y_1_test, y_1_pred)
print('MLP Regressor for X_1')
print('MAE:', mae1)
print('MSE:', mse1)
print('R^2:', r_squared1)

# Menghitung metrik evaluasi untuk y2
mae2 = mean_absolute_error(y_2_test, y_2_pred)
mse2 = mean_squared_error(y_2_test, y_2_pred)
r_squared2 = r2_score(y_2_test, y_2_pred)
print("")
print('MLP Regressor for X_2')
print('MAE:', mae2)
print('MSE:', mse2)
print('R^2:', r_squared2)

import os
import pickle

# Ganti path dengan direktori yang Anda inginkan
directory = "C:/Users/andri/Documents/Andri Martin/Coding/Prediksi/models"

if not os.path.exists(directory):
    os.makedirs(directory)

# Path untuk menyimpan kedua model dalam satu file
path = "C:/Users/andri/Documents/Andri Martin/Coding/Prediksi/models/realestate.pkl"

# Simpan model linear regression (reg_1) ke dalam file pickle
with open(path, "wb") as file:
    models = {
        "linear_regression": reg_1,
        "mlp_regressor": regressor_1
    }
    pickle.dump(models, file)

print("Models saved successfully.")

import pickle

# Muat model dari file pickle
with open('C:/Users/andri/Documents/Andri Martin/Coding/Prediksi/models/realestate.pkl', 'rb') as file:
    loaded_models = pickle.load(file)

# Mengakses model-model yang dimuat
reg_1 = loaded_models["linear_regression"]
regressor_1 = loaded_models["mlp_regressor"]

import numpy as np

# Fungsi untuk memprediksi harga rumah dengan model (linear regression atau MLP)
def predict_price(model, house_age, dist_to_MRT, conv_store):
    # Persiapkan data input dengan format yang sesuai dengan model
    input_data = np.array([[house_age, dist_to_MRT, conv_store]])

    # Lakukan prediksi dengan model yang sudah dilatih (model)
    predicted_price = model.predict(input_data)

    return predicted_price[0]

df_5

# Meminta input dari pengguna
house_age = float(input("Masukkan usia rumah: "))
dist_to_MRT = float(input("Masukkan jarak ke stasiun MRT: "))
conv_store = int(input("Masukkan jumlah toko konvenien di sekitar: "))

# Menggunakan model linear regression (reg_1)
predicted_price_linear = predict_price(reg_1, house_age, dist_to_MRT, conv_store)
print("Prediksi Harga Rumah (Linear Regression):", predicted_price_linear)

# Menggunakan model MLP (regressor_1)
predicted_price_mlp = predict_price(regressor_1, house_age, dist_to_MRT, conv_store)
print("Prediksi Harga Rumah (MLP):", predicted_price_mlp)