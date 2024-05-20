import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import configparser
from collections import Counter

config = configparser.ConfigParser()
config.read('config.ini')

def predict_satisfaction(today_class, satis):
    pred_satis = today_class - satis
    satis_string = ''
    if pred_satis >= 2:
        satis_string = "VERY_HOT"
    elif pred_satis >= 1 and pred_satis < 2:
        satis_string = "HOT"    
    elif pred_satis >= -1 and pred_satis < 1:
        satis_string = "GOOD"
    elif pred_satis >= -2 and pred_satis < -1:
        satis_string = "COLD"
    else:
        satis_string = "VERY_COLD"
    return satis_string

def find_similar_combinations(target_ids, df):
    target_ids_list = target_ids.split(', ')
    target_ids_list = [id for id in target_ids_list if int(id) <= 33]
    target_ids_counter = Counter(target_ids_list)
    df['옷 id counter'] = df['옷 id'].apply(lambda x: Counter(x.split(', ')))
    df['intersection'] = df['옷 id counter'].apply(lambda x: sum((target_ids_counter & x).values()))
    similar_combinations = df[df['intersection'] > 0]
    return similar_combinations

def adjust_satisfaction_by_item_count(target_ids, similar_ids):
    target_ids_counter = Counter([id for id in target_ids.split(', ') if int(id) <= 33])
    similar_ids_counter = Counter([id for id in similar_ids.split(', ') if int(id) <= 33])
    adjustment_value = sum(target_ids_counter.values()) - sum(similar_ids_counter.values())
    return adjustment_value

def exact_same(target_ids, df):
    target_ids_list = target_ids.split(', ')
    target_ids_counter = Counter(target_ids_list)
    df['옷 id counter'] = df['옷 id'].apply(lambda x: Counter(x.split(', ')))
    exact_same_df = df[df['옷 id counter'] == target_ids_counter]
    return exact_same_df

user_id, today, items_id, thicknesses = sys.argv[1:]
user_id = int(user_id)
today = float(today)

# csv 파일을 dataframe으로 변환
df_outfit = pd.read_csv(config.get('FilePaths', 'outfit'))
df_weather = pd.read_csv(config.get('FilePaths', 'weather'), encoding='cp949')
# 필요한 columns만 추출
df_outfit = df_outfit[['userId', '상의', '아우터', '하의', '신발', '액세서리', '작성일']].copy()
df_temp = df_weather[['일시', '평균기온(°C)']].copy()

# '작성일'과 '일시' 열을 datetime 형식으로 변환
df_outfit['작성일'] = pd.to_datetime(df_outfit['작성일'], format='%Y년 %m월 %d일')
df_temp['일시'] = pd.to_datetime(df_temp['일시'])

# 두 dataframe을 날짜를 기준으로 병합
df_merged = pd.merge(df_outfit, df_temp, left_on='작성일', right_on='일시').drop('일시', axis=1)

# 평균기온(°C) column의 최대값과 최솟값
max_temp = df_merged['평균기온(°C)'].max()
min_temp = df_merged['평균기온(°C)'].min()

def create_bins(min_temp, max_temp, step=5):
    bins = np.arange(min_temp, max_temp, step).tolist()
    bins = np.round(bins, 1).tolist()
    bins = [-np.inf] + bins + [np.inf]
    return bins

bins = create_bins(min_temp, max_temp)

satisfaction_base = config.get('FilePaths', 'satisfaction')

labels=np.arange(1, (max_temp-min_temp)//5+3)
today_class = pd.cut([today], bins=bins, labels=labels).astype('float64')
path = f'{satisfaction_base}male/user_{user_id}/satifaction.csv'

df_satisfaction = pd.read_csv(path)

target_ids = items_id[1:-1]

exact_same_df = exact_same(target_ids, df_satisfaction)
if len(exact_same_df) > 0:
    item_satis = exact_same_df.iloc[0]['예측값']
    '''print(f'today_class: {today_class}')
    print(f'item_satis: {item_satis}')
    print(f'exact_same: {exact_same_df.iloc[0]["옷 id"], predict_satisfaction(today_class, item_satis)}')
    print(f'{predict_satisfaction(today_class, item_satis)}')'''
    exit()

similar_combinations = find_similar_combinations(target_ids, df_satisfaction)

# 가장 비슷한 조합을 찾기 위해 'intersection' 열을 기준으로 내림차순 정렬
similar_combinations = similar_combinations.sort_values(by='intersection', ascending=False)

# 가장 비슷한 조합의 예측값을 가져옴
if len(similar_combinations) > 0:
    item_satis = similar_combinations.iloc[0]['예측값']
    '''print(f'today_class: {today_class}')
    print(f'item_satis: {item_satis}')
    print(f'similar_combinations: {similar_combinations.iloc[0]["옷 id"], predict_satisfaction(today_class, similar_combinations.iloc[0]["예측값"])}')'''
else:
    print("No similar combinations found.")

# 가장 비슷한 조합의 예측값을 가져옴
if len(similar_combinations) > 0:
    item_satis = similar_combinations.iloc[0]['예측값']
    similar_ids = similar_combinations.iloc[0]['옷 id']
    adjustment_value = adjust_satisfaction_by_item_count(target_ids, similar_ids)
    '''print(f'adjustment_value: {adjustment_value}')'''
    adjusted_satis = item_satis - 1.5 * adjustment_value
    '''print(f'adjusted_satis: {adjusted_satis}')'''
else:
    print("No similar combinations found.")
    adjusted_satis = item_satis

print(f'{predict_satisfaction(today_class, adjusted_satis)}')



