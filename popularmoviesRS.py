# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:05:11 2020

@author: user
"""
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/rating.csv', sep=';')
movies = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/movies.csv', sep=';', encoding='latin-1')

merged = ratings.merge(movies, left_on= 'movieId', right_on='movieId', suffixes = ['_user', ''], sort=True)
merged = merged[['title','userId','rating']]

movieStats = merged.groupby('title').agg({'rating':[np.size,np.mean]})

popularMovies = movieStats['rating']['size'] >= 100
#movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:20]
print(movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:20])