# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:48:08 2020

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

movieRatings = merged.pivot_table(index=['title'], columns=['userId'], values = 'rating')

#print(type(movieRatings))
#movieRatings.replace({np.nan:0}, regex=True, inplace=True)
movieRatings.fillna(0, inplace=True)

model_knn = NearestNeighbors(algorithm='brute', metric='cosine')
model_knn.fit(movieRatings.values)

index_of_movie = 2

distances, indices = model_knn.kneighbors(movieRatings.iloc[index_of_movie, :].values.reshape(1,-1),n_neighbors=6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:'.format(movieRatings.index[index_of_movie]))
    else:
        print('{0} : {1}, with distance of {2}:'.format(i, movieRatings.index[indices.flatten()[i]], distances.flatten()[i]))


