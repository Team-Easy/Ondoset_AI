
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import re
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import os
import sys

num_features, iterations, learning_rate, lambda_, count_weight = sys.argv[1:]
num_features = int(num_features)
iterations = int(iterations)
learning_rate = float(learning_rate)
lambda_ = float(lambda_)
count_weight = float(count_weight)
print(num_features, iterations, learning_rate, lambda_, count_weight)

# csv 파일을 dataframe으로 변환
df_outfit = pd.read_csv('../data/outfit(male)/outfit(male).csv')
df_weather = pd.read_csv('../data/2022-08-01_to_2024-04-30.csv', encoding='cp949')
# 필요한 columns만 추출
df_outfit = df_outfit[['userId', '상의', '아우터', '하의', '신발', '액세서리', '작성일']].copy()
df_temp = df_weather[['일시', '평균기온(°C)']].copy()

# '작성일'과 '일시' 열을 datetime 형식으로 변환
df_outfit['작성일'] = pd.to_datetime(df_outfit['작성일'], format='%Y년 %m월 %d일')
df_temp['일시'] = pd.to_datetime(df_temp['일시'])

# 두 dataframe을 날짜를 기준으로 병합
df_merged = pd.merge(df_outfit, df_temp, left_on='작성일', right_on='일시').drop('일시', axis=1)


# '상의', '아우터', '하의', '신발', '엑세서리' 열의 결측값을 '~ 없음'으로 대체
columns = ['상의', '아우터', '하의', '신발', '액세서리']
df_notnull = df_merged.copy()
for column in columns:
    df_notnull[column] = df_merged[column].fillna(column + ' 없음')

# 2가 붙은 단어를 두 번 반복하는 함수
def duplicate_word(text):
    words = text.split(', ')
    for i, word in enumerate(words):
        if '2' in word:
            words[i] = word.replace('2', '') + ', ' + word.replace('2', '')
    return ', '.join(words)

# 2가 붙은 단어를 두 번 반복한 dataframe df_dup 생성
df_dup = df_notnull.copy()
for column in columns:
    df_dup[columns] = df_notnull[columns].map(duplicate_word)


# 옷의 조합 컬럼 생성 (상의, 아우터, 하의, 신발, 엑세서리의 각 값들을 하나의 문자열로 조합하여 하나의 컬럼으로 만듦)
df_combination = df_dup.copy()
df_combination['옷 조합'] = df_dup['상의'] + ', ' + df_dup['아우터'] + ', ' + df_dup['하의'] + ', ' + df_dup['신발'] + ', ' + df_dup['액세서리']
df_combination.drop(columns=['상의', '아우터', '하의', '신발', '액세서리'], inplace=True)


# 쉼표를 기준으로 텍스트를 나누는 함수
def comma_tokenizer(s):
    return s.split(', ')

vectorizer = CountVectorizer(tokenizer=comma_tokenizer)

O = vectorizer.fit_transform(df_combination['옷 조합'])

# multi-hot encoding된 데이터를 numpy array로 변환
df_encoded = pd.DataFrame(O.toarray().tolist(), columns=vectorizer.get_feature_names_out())
npa = np.array(df_encoded)
'''npa.shape'''

# 값이 2 이상인 행의 인덱스
rows_with_value_2 = df_encoded[(df_encoded >= 2).any(axis=1)]

# 값이 2 이상인 열의 이름을 찾습니다.
columns_with_value_over_2 = df_encoded.columns[(df_encoded >= 2).any()]

# 특정 행에 대해 이를 기록합니다.
record = df_encoded.loc[rows_with_value_2.index, columns_with_value_over_2]


# numpy array를 list로 변환 후 clothes_combination 컬럼에 대입
df_combination['옷 조합'] = npa.tolist()

# multi-hot encoding된 데이터를 다시 텍스트로 변환
df_combination['옷 조합'] = vectorizer.inverse_transform(npa)

# 하나의 문자열로 변환
df_combtest = df_combination.copy()
df_combtest['옷 조합'] = df_combination['옷 조합'].apply(lambda x: ', '.join(map(str, x)))

# multi-hot encoding의 값이 2 이상인 경우, 해당 단어를 두 번 반복
for i in record.index:
    old_value = df_combtest.loc[i, '옷 조합']
    for col in record.columns:
        if record.loc[i, col] >= 2:
            old_value = old_value.replace(col, col + ', ' + col)
    df_combtest.loc[i, '옷 조합'] = old_value

# 평균기온(°C) column의 최대값과 최솟값
max_temp = df_combtest['평균기온(°C)'].max()
min_temp = df_combtest['평균기온(°C)'].min()

df_limit = df_combtest.copy()
# 평균기온(°C) column을 5도 간격으로 범주화하여 0, 1, 2, ...로 변환
bins=np.round(np.arange(min_temp -5, max_temp+5, 5), 1)
labels=np.arange(0, (max_temp-min_temp)//5+2)
df_limit['평균기온(°C)'] = pd.cut(df_limit['평균기온(°C)'], bins=bins, labels=labels)

# '평균기온(°C)'의 각 범주를 고려하여 데이터를 분할
train_data = []
val_data = []
test_data = []
# 각 user별로 온도 범주의 데이터가 적은 경우 기록
user_category_not_valid = {}

for user in df_limit['userId'].unique():
    for category in df_limit['평균기온(°C)'].unique():
        category_data = df_limit[(df_limit['평균기온(°C)'] == category) & (df_limit['userId'] == user)]
        
        if category_data.shape[0] < 20:
            if user not in user_category_not_valid:
                user_category_not_valid[user] = [category]
            else:
                user_category_not_valid[user].append(category)
            train_data.append(category_data)
            continue

        # 먼저 전체 데이터의 50%를 훈련 데이터로 분할
        train, temp = train_test_split(category_data, test_size=0.5, random_state=42)
        
        # 남은 데이터를 반으로 나누어 검증 데이터와 테스트 데이터로 분할
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        train_data.append(train)
        val_data.append(val)
        test_data.append(test)

# 각 데이터 세트를 하나의 DataFrame으로 병합
train_data_df = pd.concat(train_data)
val_data_df = pd.concat(val_data)
test_data_df = pd.concat(test_data)

# 평균기온 column을 범주형으로 변경
train_data_df['평균기온(°C)'] = train_data_df['평균기온(°C)'].astype('float64')
val_data_df['평균기온(°C)'] = val_data_df['평균기온(°C)'].astype('float64')
test_data_df['평균기온(°C)'] = test_data_df['평균기온(°C)'].astype('float64')

# pivot_table을 이용한 user-item matrix 생성
train_data_df_value = train_data_df.copy()
train_data_df_value['평균기온(°C)'] = train_data_df_value['평균기온(°C)'].astype('float32')
UI_temp = train_data_df_value.pivot_table(index='userId', columns='옷 조합', values='평균기온(°C)', fill_value=0)

UI_val = UI_temp.copy()
# UI_val의 값을 모두 0으로 초기화
UI_val = UI_val.map(lambda x: 0.0)
for user in UI_temp.index:
    for item in UI_temp.columns:
        # validation에 해당 user-item이 있는 경우 해당 user-item의 평균을 기록
        if item in val_data_df[val_data_df['userId'] == user]['옷 조합'].values:
            UI_val.loc[user, item] = val_data_df[(val_data_df['userId'] == user) & (val_data_df['옷 조합'] == item)]['평균기온(°C)'].mean()

UI_test = UI_temp.copy()
# UI_test의 값을 모두 0으로 초기화
UI_test = UI_test.map(lambda x: 0.0)
for user in UI_temp.index:
    for item in UI_temp.columns:
        # test에 해당 user-item이 있는 경우 해당 user-item의 평균을 기록
        if item in test_data_df[test_data_df['userId'] == user]['옷 조합'].values:
            UI_test.loc[user, item] = test_data_df[(test_data_df['userId'] == user) & (test_data_df['옷 조합'] == item)]['평균기온(°C)'].mean()

# pivot_table을 이용한 user_item_count matrix
UI_count = train_data_df.pivot_table( index='userId', columns='옷 조합', aggfunc='size', fill_value=0)
# 해당 user의 총 예제 개수로 각각의 row를 나눔
UI_count_div = UI_count.div(UI_count.sum(axis=1), axis=0)

# user-item matrix에 기록된 값이 존재하는 경우 1, 아닌 경우 0으로 변환하여 R_df에 기록
R_df = UI_temp.map(lambda x: 1 if x != 0 else 0)
R_np = np.array(R_df)
R_np.sum(axis=0)

# 각 열의 합이 2 이상(여러 유저가 해당 옷 조합을 선택한 경우)인 열을 찾음
columns_with_sum_over_2 = R_df.columns[R_df.sum() >= 2]

# CF를 위한 초기값 설정
Y = np.array(UI_temp) 
Y = Y.T
count = np.array(UI_count_div)
count = count.T
print(Y.shape)
R = Y != 0 
n_u = Y.shape[1]
n_o = Y.shape[0]


# validation, test
Y_val = np.array(UI_val)
Y_val = Y_val.T
Y_test = np.array(UI_test)
Y_test = Y_test.T


# 기록이 존재하는 값의 평균을 구함
o_sum = Y.sum(axis=1)
o_count = R.sum(axis=1)
o_mean = o_sum / o_count
o_mean = o_mean.reshape(-1, 1)

Y_stand = Y - (o_mean * R)
Y_val_stand = Y_val - (o_mean * (Y_val != 0))
Y_test_stand = Y_test - (o_mean * (Y_test != 0))


def cofi_cost_func_v(O, U, b, Y, R, lambda_):
    j = (tf.linalg.matmul(O, tf.transpose(U)) + b - Y )*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(O**2) + tf.reduce_sum(U**2))
    return J

# user, outfit의 수
n_o, n_u = Y.shape


# (U,O)를 초기화하고 tf.Variable로 등록하여 추적
tf.random.set_seed(1234) # for consistent results
U = tf.Variable(tf.random.normal((n_u,  num_features),dtype=tf.float64),  name='U')
O = tf.Variable(tf.random.normal((n_o, num_features),dtype=tf.float64),  name='O')
b = tf.Variable(tf.random.normal((1,          n_u),   dtype=tf.float64),  name='b')

# optimizer 초기화
optimizer = keras.optimizers.Adam(learning_rate = learning_rate)

J = cofi_cost_func_v(O, U, b, Y_stand, R, 1.5)

def metrics(O, U, b, o_mean, count, count_weight, df, UI_temp, labels, user_category_not_valid, isTrain=False) :
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
            
            if not isTrain:
                # user i에 대한 예측을 파일로 저장
                os.makedirs(f'../data/predictions/male/user_{i+1}', exist_ok=True)
                # Save predictions to file in user's directory
                with open(f'../data/predictions/male/user_{i+1}/predictions_{category}.txt', 'w') as f:
                    for item in predict:
                        f.write("%s\n" % item)
            
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

history = []

for iter in range(iterations):
    # TensorFlow의 GradientTape 사용
    # 연산을 기록하여 cost에 대한 gradient를 자동으로 계산
    with tf.GradientTape() as tape:

        # cost 계산 (forward pass included in cost)
        cost_value = cofi_cost_func_v(O, U, b, Y_stand, R, lambda_)

    # GradientTape를 통해 자동 미분
    # loss에 대한 trainable parameter의 gradient를 계산
    grads = tape.gradient( cost_value, [O,U,b] )

    # optimizer를 사용하여 trainable parameter를 업데이트
    optimizer.apply_gradients( zip(grads, [O,U,b]) )

    # Log periodically.
    if (iter + 1) % 20 == 0 or iter == 0:
        train_loss = cost_value.numpy()
        history.append({'type': 'train_loss', 'epoch': iter + 1, 'value': train_loss})

        val_loss = cofi_cost_func_v(O, U, b, Y_val_stand, R, lambda_).numpy()
        history.append({'type': 'validation_loss', 'epoch': iter + 1, 'value': val_loss})

        precision, recall, f1_score = metrics(O, U, b, o_mean, count, count_weight, val_data_df, UI_temp, labels, user_category_not_valid, isTrain=True)
        history.append({'type': 'validation precision', 'epoch': iter + 1, 'value': precision})
        history.append({'type': 'validation recall', 'epoch': iter + 1, 'value': recall})
        history.append({'type': 'validataion f1_score', 'epoch': iter + 1, 'value': f1_score})

print(history)

def save_variables_optimizer(variables, optimizer, filename):
    checkpoint = tf.train.Checkpoint(variables=variables, optimizer=optimizer)
    checkpoint.save(filename)


# 훈련된 tf.Variable 파일로 저장
model_version = '1.0'
checkpoint_path = f'../model/CF/{model_version}/'
os.makedirs(checkpoint_path, exist_ok=True)

save_variables_optimizer({"O": O, "U": U, "b": b}, optimizer,  checkpoint_path+"parameters.ckpt")

# UI_temp및 UI_count_div를 저장
UI_temp.to_csv(checkpoint_path + 'UI_temp.csv')
UI_count_div.to_csv(checkpoint_path + 'UI_count_div.csv')

# loss 측정을 위한 test 데이터 저장
test_for_loss = pd.DataFrame(Y_test_stand)
test_for_loss.to_csv(checkpoint_path + 'test_for_loss.csv')

# metric을 계산하기 위한 데이터 저장
test_data_df.to_csv(checkpoint_path + 'test_data_df.csv')

# user_category_not_valid 및 unique category 저장
user_category_not_valid_df = pd.DataFrame(user_category_not_valid.items(), columns=['userId', 'category'])
user_category_not_valid_df.to_csv(checkpoint_path + 'user_category_not_valid.csv')
category = df_limit['평균기온(°C)'].unique()

# 범주 및 해당 범주의 온도 범위를 확인하여 category_df의 columns에 저장
categorized_temperature = pd.cut([-1.8], bins=bins, labels=labels)
l_temp = min_temp -5
category_values = []
for i in range(len(bins) - 1):
    category_values.append(f'({bins[i]}, {bins[i+1]}]')
category_df = pd.DataFrame(columns=categorized_temperature.categories, data=[category_values])
category_df.to_csv(checkpoint_path + 'category.csv')
