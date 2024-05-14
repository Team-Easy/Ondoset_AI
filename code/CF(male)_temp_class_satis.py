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

user_id, today, items_id = sys.argv[1:]
user_id = int(user_id)
today = float(today)

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

# 평균기온(°C) column의 최대값과 최솟값
max_temp = df_merged['평균기온(°C)'].max()
min_temp = df_merged['평균기온(°C)'].min()

bins=np.round(np.arange(min_temp -5, max_temp+5, 5), 1)
labels=np.arange(0, (max_temp-min_temp)//5+2)
today_class = pd.cut([today], bins=bins, labels=labels).astype('float64')

path = f'../data/satisfaction/CF/male/user_{user_id}/satifaction.csv'

df_satisfaction = pd.read_csv(path)

item_satis = df_satisfaction[df_satisfaction['옷 id'] == "2, 17, 28, 36, 45, 51"]['예측값'].values
pred_satis = today_class - item_satis

if pred_satis >= 2:
    print("VERY_HOT")
elif pred_satis >= 1 and pred_satis < 2:
    print("HOT")
elif pred_satis >= -1 and pred_satis < 1:
    print("NORMAL")
elif pred_satis >= -2 and pred_satis < -1:
    print("COLD")
else:
    print("VERY_COLD")

