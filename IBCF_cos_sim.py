# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:44:12 2020

@author: user
"""

import pandas as pd
import numpy as np

from  sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/rating.csv', sep=';')
movies = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/movies.csv', sep=';', encoding='latin-1')

merged = ratings.merge(movies, left_on= 'movieId', right_on='movieId', sort=True)

merged = merged[['userId', 'title', 'rating']]

movieRatings = merged.pivot_table(index=['title'], columns=['userId'], values = 'rating')

#print(type(movieRatings))
#movieRatings.replace({np.nan:0}, regex=True, inplace=True)
movieRatings.fillna(0, inplace=True)

item_similarity = cosine_similarity(movieRatings)

item_sim_df = pd.DataFrame(item_similarity, index=movieRatings.index, columns = movieRatings.index)

def sim_movies_to(title):
    count = 1
    print('Similar movies to {} are :'.format(title))
    for item in item_sim_df.sort_values(by=title, ascending=False).index[1:11]:
        print('No. {} : {}'.format(count,item))
        count +=1
        
sim_movies_to('Toy Story (1995)')
sim_movies_to('Father of the Bride Part II (1995)')