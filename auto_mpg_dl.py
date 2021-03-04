#module Name : auto_mpg_dl.py
# 1. mpg:           continuous
# 2. cylinders:     multi-valued discrete
# 3. displacement:  continuous
# 4. horsepower:    continuous
# 5. weight:        continuous
# 6. acceleration:  continuous
# 7. model year:    multi-valued discrete
# 8. origin:        multi-valued discrete
# 9. car name:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score,precision_score,recall_score, roc_curve, classification_report,precision_recall_curve
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler, Binarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

import warnings
warnings.filterwarnings(action="ignore")

np.random.seed(121)
tf.random.set_seed(121)

def CHART_PLOT_HISTORY(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))

    plt.subplot(2,1,1)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(hist['epoch'], hist['mse'],    label='mse')
    plt.plot(hist['epoch'], hist['val_mse'],label = 'val_mse')
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'], hist['loss'],    label='loss')
    plt.plot(hist['epoch'], hist['val_loss'],label = 'val_loss')
    plt.legend()
    plt.show()

cols = ['mpg','cylinder','display','power','weight','accel','year','origin','name']
df = pd.read_csv("auto-mpg.data",
                 na_values='?',
                 delimiter=r"\s+",
                 names=cols,
                 )
#------------------- Dataframe 확인
print(df.shape)     #(398, 9)
print(df.info())    #결측 X, Object X
print(df.head())

print(df.describe().T)

# 1.  결측처리
# power     392 non-null    float64
df["power"] = df["power"].fillna(df["power"].mean())
print(df.isna().sum())
print(df["power"][:10])

# 2. name - object
# print(df["name"].nunique())  #305
df.drop("name", inplace=True, axis=1)

# origin  : 원핫인코딩 --> 1. usa  2. eu  3. jp
# target : mpg

# ============ 분석 : 결측X, ObjectX
# sns.heatmap(df.corr(), annot=True, fmt=".3f")
# plt.show()

def BUILD_EVAL(X_data, y, str=None, epoch=100):
    model = Sequential()
    model.add(Dense(32, input_dim=X_data.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    # print(model.summary())

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse', 'mae'])

    from keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    res = model.fit(X_data, y,
                    validation_split=0.1,
                    epochs=epoch,
                    callbacks=[early_stop])

    # res.histroy['val_loss']
    # loss: 18.5499 - mse: 18.5499 - mae: 3.3252 - val_loss: 44.7556 - val_mse: 44.7556 - val_mae: 4.9777
    loss, mse, mae = model.evaluate(X_data, y)
    print(str, "epoch 100회 평균 loss {:.4f}  mse {:.4f}  rmse {:.4f} mae {:.4f}".format(loss, mse, np.sqrt(mse), mae))
    # CHART_PLOT_HISTORY(res)


y = df["mpg"]
X = df.drop("mpg", axis=1)
BUILD_EVAL(X, y, "----1차---")

# ------------------------------------- PCA : 다중공선
pca_col = ['cylinder','display','power','weight']

# ------------------------------------- one-hot encoding
# for c in df.columns:
#     print(c, df[c].nunique())
# REF : https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
#
# lenc = LabelEncoder()
# for c in enc_col:
#     print(X[c].to_numpy().reshape(-1,1)[:5])
#     X[c] = lenc.fit_transform(X[c].to_numpy().reshape(-1,1))
#
# oh = OneHotEncoder()
# X = oh.fit_transform(X[enc_col])
# X = oh.fit_transform(X[['door']])

enc_col = ['cylinder', 'year', 'origin']
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# pd.get_dummies() //결측 + 글자->숫자(LabelEncoder)  OneHotEncoder(001)
oh_X = pd.get_dummies(X, columns=enc_col)
BUILD_EVAL(oh_X, y, "----인코딩---")



# ------------------------------------- np.log1p scaling
log_col = ['display','power','weight']
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit_transform()
# BUILD_EVAL(sc_X)

for c in log_col:
    oh_X[c] = np.log1p(np.array(oh_X[c]))
oh_y = np.log1p(np.array(df["mpg"]))

BUILD_EVAL(oh_X, oh_y, "----스케일링---")