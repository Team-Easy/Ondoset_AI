import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def extract_columns(df_outfit, df_weather):
    df_outfit = df_outfit[['userId', '상의', '아우터', '하의', '신발', '액세서리', '작성일']].copy()
    df_temp = df_weather[['일시', '평균기온(°C)']].copy()
    return df_outfit, df_temp

def convert_to_datetime(df_outfit, df_temp):
    df_outfit['작성일'] = pd.to_datetime(df_outfit['작성일'], format='%Y년 %m월 %d일')
    df_temp['일시'] = pd.to_datetime(df_temp['일시'])
    return df_outfit, df_temp

def merge_dataframes(df_outfit, df_temp):
    df_merged = pd.merge(df_outfit, df_temp, left_on='작성일', right_on='일시').drop('일시', axis=1)
    return df_merged

def fill_na(df_merged):
    columns = ['상의', '아우터', '하의', '신발', '액세서리']
    df_notnull = df_merged.copy()
    for column in columns:
        df_notnull[column] = df_merged[column].fillna(column + ' 없음')
    return df_notnull

def duplicate_word(text):
    words = text.split(', ')
    for i, word in enumerate(words):
        if '2' in word:
            words[i] = word.replace('2', '') + ', ' + word.replace('2', '')
    return ', '.join(words)

def duplicate_df(df_notnull):
    columns = ['상의', '아우터', '하의', '신발', '액세서리']
    df_dup = df_notnull.copy()
    for column in columns:
        df_dup[column] = df_notnull[column].map(duplicate_word)
    return df_dup

def create_combination(df_dup):
    df_combination = df_dup.copy()
    df_combination['옷 조합'] = df_dup['상의'] + ', ' + df_dup['아우터'] + ', ' + df_dup['하의'] + ', ' + df_dup['신발'] + ', ' + df_dup['액세서리']
    df_combination.drop(columns=['상의', '아우터', '하의', '신발', '액세서리'], inplace=True)
    return df_combination

def comma_tokenizer(s):
    return s.split(', ')

def vectorize(df_combination):
    vectorizer = CountVectorizer(tokenizer=comma_tokenizer)
    O = vectorizer.fit_transform(df_combination['옷 조합'])
    df_encoded = pd.DataFrame(O.toarray().tolist(), columns=vectorizer.get_feature_names_out())
    return df_encoded, vectorizer

def create_bins(min_temp, max_temp, step=5):
    bins = np.arange(min_temp, max_temp, step).tolist()
    bins = np.round(bins, 1).tolist()
    bins = [-np.inf] + bins + [np.inf]
    return bins

def process_data(df_outfit, df_weather):
    df_outfit, df_temp = extract_columns(df_outfit, df_weather)
    df_outfit, df_temp = convert_to_datetime(df_outfit, df_temp)
    df_merged = merge_dataframes(df_outfit, df_temp)
    df_notnull = fill_na(df_merged)
    df_dup = duplicate_df(df_notnull)
    df_combination = create_combination(df_dup)
    df_encoded, vectorizer = vectorize(df_combination)
    npa = np.array(df_encoded)
    df_combination['옷 조합'] = npa.tolist()
    df_combination['옷 조합'] = vectorizer.inverse_transform(npa)
    df_combtest = df_combination.copy()
    df_combtest['옷 조합'] = df_combination['옷 조합'].apply(lambda x: ', '.join(map(str, x)))
    max_temp = df_combtest['평균기온(°C)'].max()
    min_temp = df_combtest['평균기온(°C)'].min()
    df_limit = df_combtest.copy()
    bins = create_bins(min_temp, max_temp)
    labels=np.arange(1, (max_temp-min_temp)//5+3)
    df_limit['평균기온(°C)'] = pd.cut(df_limit['평균기온(°C)'], bins=bins, labels=labels)
    return df_limit, bins, labels