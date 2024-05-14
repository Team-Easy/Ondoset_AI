import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_similarity

user_id,= sys.argv[1:]
user_id = int(user_id)

latent_factor = pd.read_csv('../data/similarity/User_latent_factors.csv')

latent_columns = latent_factor.columns[1:]

# userId의 latent factor 선택
user_factor = latent_factor[latent_factor['userId'] == user_id]

# 다른 userId의 latent factor 선택
other_factors = latent_factor[latent_factor['userId'] != user_id]

# user_factor와 ohter_factors의 cosine similarity를 계산
similarity = cosine_similarity(user_factor[latent_columns], other_factors[latent_columns])

# similarity 상위 10개의 userId를 배열로 저장
top_similar_arr = []
similarity = similarity[0]
top_similar_userId = other_factors.iloc[similarity.argsort()[::-1][:10]]['userId']
for i in top_similar_userId.values :
    top_similar_arr.append(i)
print(top_similar_arr)
