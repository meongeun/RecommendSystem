# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:25:17 2020

@author: user
"""

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/rating.csv', sep=';')
movies = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/movies.csv', sep=';', encoding='latin-1')

merged = ratings.merge(movies, left_on= 'movieId', right_on='movieId', sort=True)

merged = merged[['userId', 'title', 'rating']]
print(merged)

movieRatings = merged.pivot_table(index=['userId'], columns=['title'], values = 'rating')

#print(type(movieRatings))
#movieRatings.replace({np.nan:0}, regex=True, inplace=True)
movieRatings.fillna(0, inplace=True)

model_knn = NearestNeighbors(algorithm='brute', metric='cosine')
model_knn.fit(movieRatings.values)

user = 2

distances, indices = model_knn.kneighbors(movieRatings.iloc[user-1, :].values.reshape(1,-1),n_neighbors=7)

import operator
best = []
movieRatings = movieRatings.T
for i in indices.flatten():
    if i != user-1:
        max_score = movieRatings.loc[:,i+1].max()
        best.append(movieRatings[movieRatings.loc[:,i+1] == max_score].index.tolist())
    #print(movieRatings[movieRatings.loc[:,i] == max_score].index)

user_seen_movies = movieRatings[movieRatings.loc[:, user]> 0].index.tolist()

for i in range(len(best)):
    for j in best[i]:
        if j in user_seen_movies:
            best[i].remove(j)
        
most_common = {}
for i in range(len(best)):
    for j in best[i]:
        if j in most_common:
            most_common[j] +=1
        else:
            most_common[j] =1
#print(most_common.items())
sorted_list = sorted(most_common.items(), key = operator.itemgetter(1), reverse=True)

#print(sorted_list[:5])