import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import keras
import os
import sys
import configparser
import preprocess_ow

num_features, iterations, learning_rate, lambda_, count_weight = sys.argv[1:]
num_features = int(num_features)
iterations = int(iterations)
learning_rate = float(learning_rate)
lambda_ = float(lambda_)
count_weight = float(count_weight)

# 경로 설정 파일
config = configparser.ConfigParser()
config.read('config.ini')

# csv 파일을 dataframe으로 변환
df_outfit = pd.read_csv(config.get('FilePaths', 'outfit'))
df_weather = pd.read_csv(config.get('FilePaths', 'weather'), encoding='cp949')

df_limit, bins, labels = preprocess_ow.process_data(df_outfit, df_weather)

# pivot_table을 이용한 user-item matrix 생성
train_data_df_value = df_limit.copy()
train_data_df_value['평균기온(°C)'] = train_data_df_value['평균기온(°C)'].astype('float32')
UI_temp = train_data_df_value.pivot_table(index='userId', columns='옷 조합', values='평균기온(°C)', fill_value=0)

# 첫번째 행의 모든 값을 0으로 변경
UI_temp.iloc[0] = 0

# pivot_table을 이용한 user_
UI_count = df_limit.pivot_table( index='userId', columns='옷 조합', aggfunc='size', fill_value=0.0)

# 첫번째 행의 모든 값을 1으로 변경
UI_count.iloc[0] = 1

# 해당 user의 총 예제 개수로 각각의 row를 나눔 (여기서 해당 유저의 총 예제 개수가 100개가 안된다면 100개로 나눔)
UI_sum = UI_count.sum(axis=1)
UI_sum[UI_sum < 200] = 200  # 총 예제 개수가 100개 미만인 경우 100으로 설정
UI_count_div = UI_count.div(UI_sum, axis=0)

# UI_temp에서 상의 없음, 신발 없음, 아우터 없음, 액세서리 없음, 하의 없음 열 삭제
UI_temp = UI_temp.drop(columns=['상의 없음, 신발 없음, 아우터 없음, 액세서리 없음, 하의 없음'], axis=1)
UI_count_div =  UI_count_div.drop(columns=['상의 없음, 신발 없음, 아우터 없음, 액세서리 없음, 하의 없음'], axis=1)

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
similarity_base = config.get('FilePaths', 'similarity')
df_U = pd.DataFrame(U.numpy(), index=UI_temp.index, columns=np.ndarray.tolist(np.arange(1, num_features+1)))
os.makedirs(f'{similarity_base}', exist_ok=True)
df_U.to_csv(f'{similarity_base}/User_latent_factors.csv')

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
            
            predict_base = config.get('FilePaths', 'predict')
            
            # user i에 대한 예측을 파일로 저장
            os.makedirs(f'{predict_base}male/user_{i}', exist_ok=True)
            os.makedirs(f'{predict_base}male/debug/user_{i}', exist_ok=True)
            # Save predictions to file in user's directory
            with open(f'{predict_base}male/user_{i}/predictions_{category}.txt', 'w') as f:
                for item in predict_id:
                    f.write("%s\n" % item)   
                for item in thick:
                    f.write("%s\n" % item)   
            
            with open(f'{predict_base}male/debug/user_{i}/predictions_{category}_tag.txt', 'w') as f:
                for item in predict:
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

satisfaction_base = config.get('FilePaths', 'satisfaction')

for i in range(UI_satis_id.shape[1]):
    user_id = UI_satis_id.columns[i]
    temp = UI_satis_id.copy()
    temp = temp[[user_id]]
    temp.columns = ['예측값']
    temp.reset_index(inplace=True)
    temp.columns = ['옷 id', '예측값']
    # UI_satis의 해당하는 user_id column의 각 값에 대해 j값을 뺌
    # user i에 대한 예측을 파일로 저장
    os.makedirs(f'{satisfaction_base}male/user_{i}', exist_ok=True)
    temp.to_csv(f'{satisfaction_base}male/user_{user_id}/satifaction.csv', index=False, header=True)

# 전체 데이터 개수를 반환
print(f'{len(df_outfit)}')