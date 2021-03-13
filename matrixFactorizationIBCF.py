# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:13:25 2020

@author: user
"""

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

from  sklearn.metrics.pairwise import cosine_similarity

ratings_df = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/rating.csv', sep=';')
movies_df = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/movies.csv', sep=';', encoding='latin-1')

A_df = ratings_df.pivot_table(index=['userId'], columns=['movieId'], values='rating', aggfunc=np.max)

A_df.fillna(0, inplace=True)

A = A_df.values
#print(A.shape)
user_rating_mean = np.mean(A, axis=1)

A_normalized = A - user_rating_mean.reshape(-1,1)

U, sigma, Vt = svds(A_normalized, k=50)

sigma  = np.diag(sigma)

predicted_rating = np.dot(np.dot(U, sigma), Vt) + user_rating_mean.reshape(-1,1)

predicted_rating_df = pd.DataFrame(predicted_rating, columns= A_df.columns)
preds_df = np.transpose(predicted_rating_df)

item_similarity = cosine_similarity(preds_df)

item_sim_df = pd.DataFrame(item_similarity, index=preds_df.index, columns = preds_df.index)
# item_sim_df.columns=item_sim_df.columns.str.strip()
# print(item_sim_df)
# item_sim_df.columns = [col.strip() for col in list(item_sim_df.columns)]
def sim_movies_to(movieId):
    global movies_df
    count = 1
    movieIndex = movies_df.index[movies_df['movieId'] == movieId]+1
    print('Similar movies to {} are :'.format(movies_df.loc[movieIndex].title))
    # print(item_sim_df.sort_values(by=movieId, ascending=False))
    for item in item_sim_df.sort_values(by=movieId, ascending=False).index[1:11]:
        itemIndex = movies_df.index[movies_df['movieId'] ==item]
        print('No. {} : {}'.format(count, movies_df.loc[itemIndex].title))
        count+=1   
        
sim_movies_to(2)
sim_movies_to(5)