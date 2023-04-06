import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def trans_date(df, col_name, trans_col): #str형 날짜를 date형태로 바꿔주는 함수

    df[trans_col] = pd.to_datetime(df[col_name])

    return df[trans_col]


def MinMaxScaling(df, col_name):

    dfx = df[[col_name]]
    
    for col_name in dfx.columns:
        scaler = MinMaxScaler() #minmax스케일링
        dfx = dfx.copy()
        dfx.loc[:,col_name] = scaler.fit_transform(dfx)
    df_result = dfx[[col_name]]
    t = df_result[col_name].to_list() #데이터 프레임을 리스트로 바꿔줌 
    return t

def validation_split(x, cut_size, all_size): #train, test 데이터 분리 
    train, test = x[0:cut_size], x[cut_size:all_size]
    # print(len(train))
    # print(len(test))
    return train, test

def convert_to_matrix(data, step):
    x, y = [], []
    for i in range(len(data) - step):
        d = i + step  
        x.append(data[i:d])
        y.append(data[d])
    return np.array(x), np.array(y)

def data_reshape(train, test):
    train = np.append(train, np.repeat(train[-1], 4))
    test = np.append(test, np.repeat(test[-1], 4))

    train_x, train_y = convert_to_matrix(train, 4)
    test_x, test_y = convert_to_matrix(test, 4)

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    return [train_x, train_y, test_x, test_y]  