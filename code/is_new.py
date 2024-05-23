import pandas as pd
import sys
import configparser

userId = sys.argv[1]
userId = int(userId)

config = configparser.ConfigParser()
config.read('config.ini')

df_outfit = pd.read_csv(config.get('FilePaths', 'outfit'))

# df_outfit의 userId = 0인 row를 제외
df_outfit = df_outfit[df_outfit['userId'] != 0]

# 각 userId 기준으로 데이터의 개수를 확인
data_count = df_outfit['userId'].value_counts()

if userId in data_count.index:
    if data_count[userId] >= 50:
        print('해당 userId의 데이터가 충분함')
        print(userId)
    else :
        print('userId는 존재하지만 데이터가 부족')
        print(0)
else :
    print('userId가 존재하지 않음')
    print(0)