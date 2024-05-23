import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import sys
import json
import configparser

model_version, num_features, iterations, learning_rate, lambda_, count_weight = sys.argv[1:]
num_features = int(num_features)
iterations = int(iterations)
learning_rate = float(learning_rate)
lambda_ = float(lambda_)
count_weight = float(count_weight)

config = configparser.ConfigParser()
config.read('config.ini')

model_base = config.get('FilePaths', 'model')
# 저장된 모델 버전 및 체크포인트 경로 설정
checkpoint_path = f'{model_base}{model_version}/'

# CSV 파일 불러오기
UI_temp = pd.read_csv(checkpoint_path + 'UI_temp.csv').drop('userId', axis=1)
UI_count_div = pd.read_csv(checkpoint_path + 'UI_count_div.csv').drop('userId', axis=1)
test_for_loss = pd.read_csv(checkpoint_path + 'test_for_loss.csv')
test_data_df = pd.read_csv(checkpoint_path + 'test_data_df.csv')
user_category_not_valid_df = pd.read_csv(checkpoint_path + 'user_category_not_valid.csv')
category_df = pd.read_csv(checkpoint_path + 'category.csv')

UI_test = UI_temp.copy()
# UI_test의 값을 모두 0으로 초기화
UI_test = UI_test.map(lambda x: -0.2)
for user in UI_temp.index:
    for item in UI_temp.columns:
        # test에 해당 user-item이 있는 경우 해당 user-item의 평균을 기록
        if item in test_data_df[test_data_df['userId'] == user]['옷 조합'].values:
            UI_test.loc[user, item] = test_data_df[(test_data_df['userId'] == user) & (test_data_df['옷 조합'] == item)]['평균기온(°C)'].mean()

bins = category_df.values[0, 1:]
labels = category_df.columns[1:]

# labels의 dtype을 float64로 변경
labels = labels.astype('float64')

# CF를 위한 초기값 설정
Y = np.array(UI_temp) 
Y = Y.T
count = np.array(UI_count_div)
count = count.T
R = Y != 0 
n_u = Y.shape[1]
n_o = Y.shape[0]

# 기록이 존재하는 값의 평균을 구함
o_sum = Y.sum(axis=1)
o_count = R.sum(axis=1)
o_mean = o_sum / o_count
o_mean = o_mean.reshape(-1, 1)

Y_test = np.array(UI_test)
Y_test = Y_test.T

Y_stand = Y - (o_mean * R)
Y_test_stand = Y_test - (o_mean * (Y_test != -0.2))

# user, outfit의 수
n_o, n_u = Y.shape

# (U,O)를 초기화하고 tf.Variable로 등록하여 추적
tf.random.set_seed(1234) # for consistent results
U = tf.Variable(tf.random.normal((n_u,  num_features),dtype=tf.float64),  name='U')
O = tf.Variable(tf.random.normal((n_o, num_features),dtype=tf.float64),  name='O')
b = tf.Variable(tf.random.normal((1,          n_u),   dtype=tf.float64),  name='b')

# optimizer 초기화
optimizer = keras.optimizers.Adam(learning_rate = learning_rate)

def load_variables_optimizer(variables, optimizer, checkpoint_path):
    checkpoint = tf.train.Checkpoint(variables=variables, optimizer=optimizer)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

# 훈련된 변수 불러오기
load_variables_optimizer({"O": O, "U": U, "b": b}, optimizer,  checkpoint_path)


def cofi_cost_func_v(O, U, b, Y, R, lambda_):
    j = (tf.linalg.matmul(O, tf.transpose(U)) + b - Y )*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(O**2) + tf.reduce_sum(U**2))
    return J


def test(O, U, b, o_mean, count, count_weight, df, UI_temp, labels, user_category_not_valid) :
    # 예측을 수행하기 위해 모든 user-item에 대한 예측값을 계산
    p = np.matmul(O.numpy(), np.transpose(U.numpy())) + b.numpy()
    # user_category_not_valid에 해당하지 않는 경우에 대해 precision, recall, f1_score 계산
    # 평균을 위한 초기화
    precision_m, recall_m, f1_score_m, count_m = 0, 0, 0, 0
    for i in range(UI_temp.shape[0]):
        for category in labels:
            
            # 실제 온도
            # 평균을 적용하고 temp를 빼서 값이 작을수록 실제 온도에 가깝도록 함. 이 때 각 user-item의 사용 횟수를 가중하여 많이 사용한 item이 추천되도록 함
            pm = np.power(p + o_mean - category, 2)  -count * count_weight
            my_predictions = pm[:,i]

            # sort predictions
            ix = tf.argsort(my_predictions, direction='ASCENDING')

            df_predict = UI_temp[UI_temp.columns[ix[0:3]]].copy()
            df_predict = df_predict.round(0)
            # df_predict의 columns와 test_data_df의 '옷 조합' column을 비교하여 일치하는 경우의 개수를 계산
            predict = df_predict.columns.astype(str)
            
            if i+1 in user_category_not_valid and category in user_category_not_valid[i+1]:
                continue
            
            label = df[(df['userId'] == i+1) & (df['평균기온(°C)'] == category)]['옷 조합'].astype(str)
            # label이 UI_temp의 column에 포함되지 않는다면 제외
            label = label[label.isin(UI_temp.columns)]
            # label에 어떠한 옷 조합도 포함되지 않을 시 지표를 측정하지 않음
            if label.shape[0] == 0:
                continue
            
            count_m += 1
            precision = len(set(predict) & set(label)) / len(set(predict))
            recall = len(set(predict) & set(label)) / len(set(label))
            if precision + recall == 0: 
                f1_score = 0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)
            precision_m += precision
            recall_m += recall
            f1_score_m += f1_score
    precision_m /= count_m
    recall_m /= count_m
    f1_score_m /= count_m
    return precision_m, recall_m, f1_score_m

precision, recall, f1_score = test(O, U, b, o_mean, count, count_weight, test_data_df, UI_temp, labels, user_category_not_valid_df)

data = {'loss': [cofi_cost_func_v(O, U, b, Y_test_stand, R, lambda_).numpy().astype('float64')], 'precision': [precision], 'recall': [recall], 'f1_score': [f1_score]}
json_data = json.dumps(data)
print(json_data)


