# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:04:44 2020

@author: user
"""

import pandas as pd
import numpy as np

ratings = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/rating.csv', sep=';')
movies = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/movies.csv', sep=';', encoding='latin-1')

ts = ratings['timestamp']

ts = pd.to_datetime(ts, unit='s').dt.hour
movies['hours'] = ts

merged = ratings.merge(movies, left_on= 'movieId', right_on='movieId', suffixes = ['_user', ''])
merged = merged[['userId', 'movieId', 'genres','hours']]

merged = pd.concat([merged, merged['genres'].str.get_dummies(sep='|')], axis=1)
del merged['genres']
del merged['(no genres listed)']

def activeuserprofile(userId): 
    userprofile = merged.loc[merged['userId'] == userId]
    del userprofile['userId']
    userprofile = userprofile.groupby(['hours'], as_index=False, sort=True).sum()
    userprofile.iloc[:, 1:20] = userprofile.iloc[:, 1:20].apply(lambda x: (x - np.min(x))/(np.max(x)- np.min(x)),axis=1)
    
    return userprofile

activeuser = activeuserprofile(30)
recommend = pd.read_csv('C:/Users/user/Desktop/KME/Recommender-Python-Code/recommend.csv', sep=";")

merged = merged.drop_duplicates()

user_pref = recommend.merge(merged, left_on= 'movieId', right_on='movieId', suffixes = ['_user', ''])
# recommend.to_csv('recommend.csv', sep=';', iindex=False)    
#print(user_pref)
#print(activeuser)
product = np.dot(user_pref.iloc[:, 2:21].values, activeuser.iloc[15, 1:20].values)
preferences = np.stack((user_pref['movieId'], product), axis=-1)
df = pd.DataFrame(preferences, columns = ['movieId', 'preferences'])
result = (df.sort_values(['preferences'], ascending=False)).iloc[0:10,0]    
    








