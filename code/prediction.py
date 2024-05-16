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

# csv 파일을 dataframe으로 변환
df_outfit = pd.read_csv('/home/t24119/v1.0src/ai/data/outfit(male)/outfit(male).csv')
df_weather = pd.read_csv('/home/t24119/v1.0src/ai/data/2022-08-01_to_2024-04-30.csv', encoding='cp949')
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

# 값이 2 이상인 행의 인덱스
rows_with_value_2 = df_encoded[(df_encoded >= 2).any(axis=1)]
rows_with_value_2.index

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
'''print(max_temp, min_temp)'''

df_limit = df_combtest.copy()
# 평균기온(°C) column을 5도 간격으로 범주화하여 0, 1, 2, ...로 변환
bins=np.round(np.arange(min_temp -5, max_temp+5, 5), 1)
labels=np.arange(0, (max_temp-min_temp)//5+2)
df_limit['평균기온(°C)'] = pd.cut(df_limit['평균기온(°C)'], bins=bins, labels=labels)
'''df_limit'''

# pivot_table을 이용한 user-item matrix 생성
train_data_df_value = df_limit.copy()
train_data_df_value['평균기온(°C)'] = train_data_df_value['평균기온(°C)'].astype('float32')
UI_temp = train_data_df_value.pivot_table(index='userId', columns='옷 조합', values='평균기온(°C)', fill_value=0)

# pivot_table을 이용한 user_
UI_count = df_limit.pivot_table( index='userId', columns='옷 조합', aggfunc='size', fill_value=-2.0)
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
R = Y != 0 
n_u = Y.shape[1]
n_o = Y.shape[0]

# 기록이 존재하는 값의 평균을 구함
o_sum = Y.sum(axis=1)
o_count = R.sum(axis=1)
o_mean = o_sum / o_count
o_mean = o_mean.reshape(-1, 1)

Y_stand = Y - (o_mean * R)

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

# U의 값을 csv 파일로 저장
df_U = pd.DataFrame(U.numpy(), index=UI_temp.index, columns=np.ndarray.tolist(np.arange(1, num_features+1)))
os.makedirs('/home/t24119/v1.0src/ai/data/similarity', exist_ok=True)
df_U.to_csv('/home/t24119/v1.0src/ai/data/similarity/User_latent_factors.csv')

item_dictionary = {
    "반팔 티": 1,
    "긴팔 티": 2,
    "민소매 티": 3,
    "반팔 니트": 4,
    "니트": 5,
    "후드티": 6,
    "맨투맨": 7,
    "반팔 셔츠/블라우스": 8,
    "셔츠/블라우스": 9,
    "점프슈트": 10,
    "미니/미디원피스": 11,
    "롱원피스": 12,
    "반바지": 13,
    "데님팬츠": 14,
    "면바지": 15,
    "슬랙스": 16,
    "트레이닝/조거 팬츠": 17,
    "카고바지": 18,
    "레깅스": 19,
    "가죽 바지": 20,
    "나일론 팬츠": 21,
    "미니/미디스커트": 22,
    "롱스커트": 23,
    "집업": 24,
    "재킷": 25,
    "바람막이": 26,
    "점퍼": 27,
    "가디건": 28,
    "코트": 29,
    "조끼": 30,
    "패딩조끼": 31,
    "패딩": 32,
    "롱패딩": 33,
    "운동화": 34,
    "스니커즈/캔버스": 35,
    "구두/로퍼": 36,
    "힐": 37,
    "샌들/슬리퍼": 38,
    "레더부츠": 39,
    "어그부츠": 40,
    "레인부츠": 41,
    "패딩슈즈": 42,
    "비니": 43,
    "털 모자": 44,
    "기타 모자": 45,
    "마스크": 46,
    "머플러": 47,
    "스카프": 48,
    "장갑": 49,
    "양말": 50,
    "장목양말": 51,
    "니삭스": 52,
    "스타킹": 53,
    "상의 없음": 54,
    "아우터 없음": 55,
    "하의 없음": 56,
    "신발 없음": 57,
    "액세서리 없음": 58
    }

def to_id(item_dictionary, predict) :
    predict_result = []
    for i in predict:
        items = i.split(', ')
        predict_id = []
        for j in items:
            # 만약 items에 '없음'이라는 문자가 포함되면 continue
            if '없음' in j:
                continue
            predict_id.append(item_dictionary[j])
        # predict_id를 sort
        predict_id.sort()
        predict_result.append(predict_id)
    return predict_result

def index_id(item_dictionary, columns) :
    columns_id = []
    for i in columns:
        items = i.split(', ')
        column_id = []
        for j in items:
            # 만약 items에 '없음'이라는 문자가 포함되면 continue
            if '없음' in j:
                continue
            column_id.append(item_dictionary[j])
        # predict_id를 sort
        column_id.sort()
        # colums_id를 문자열로 변환
        column_id = ', '.join(map(str, column_id))
        columns_id.append(column_id)
    return columns_id

def predict(O, U, b, o_mean, count, count_weight, UI_temp, labels, item_dictionary) :
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
            # df_predict의 columns와 test_data_df의 '옷 조합' column을 비교하여 일치하는 경우의 개수를 계산
            predict = df_predict.columns.astype(str)
            
            predict_id = to_id(item_dictionary, predict)
            
            thick = []
            for k in predict_id:
                thick_comb = []
                for item in k:
                    thick_comb.append(-2)
                thick.append(thick_comb)
            
            # user i에 대한 예측을 파일로 저장
            os.makedirs(f'/home/t24119/v1.0src/ai/data/predictions/CF/male/user_{i+1}', exist_ok=True)
            # Save predictions to file in user's directory
            with open(f'/home/t24119/v1.0src/ai/data/predictions/CF/male/user_{i+1}/predictions_{category}.txt', 'w') as f:
                for item in predict_id:
                    f.write("%s\n" % item)   
                for item in thick:
                    f.write("%s\n" % item)          

predict(O, U, b, o_mean, count, count_weight, UI_temp, labels, item_dictionary)

def satis(O, U, b, o_mean):
    p = np.matmul(O.numpy(), np.transpose(U.numpy())) + b.numpy()
    p = p + o_mean
    return p

p = satis(O, U, b, o_mean)
p_round = np.round(p, 1)
UI_satis = pd.DataFrame(p_round, columns=UI_temp.index, index=UI_temp.columns)

# UI_satis의 각 index를 to_id 함수를 이용하여 id로 변환
UI_satis_id = UI_satis.copy()
UI_satis_id.index = index_id(item_dictionary, UI_satis_id.index.astype(str))

for i in range(UI_satis_id.shape[1]):
    user_id = UI_satis_id.columns[i]
    temp = UI_satis_id.copy()
    temp = temp[[user_id]]
    temp.columns = ['예측값']
    temp.reset_index(inplace=True)
    temp.columns = ['옷 id', '예측값']
    # UI_satis의 해당하는 user_id column의 각 값에 대해 j값을 뺌
    # user i에 대한 예측을 파일로 저장
    os.makedirs(f'/home/t24119/v1.0src/ai/data/satisfaction/CF/male/user_{i+1}', exist_ok=True)
    temp.to_csv(f'/home/t24119/v1.0src/ai/data/satisfaction/CF/male/user_{user_id}/satifaction.csv', index=False, header=True)

# 전체 데이터 개수를 반환
print(f'{len(df_outfit)}')