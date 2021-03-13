# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:28:24 2020

@author: user
"""
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

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

def recommend_movies(prediction_df, userID, movies_df, original_ratings_df, num_recommedations=5):
    user_row_number = userID-1
    sorted_user_predictions = predicted_rating_df.iloc[user_row_number].sort_values(ascending=False)
    
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = (user_data.merge(movies_df, how= 'left', left_on='movieId', right_on='movieId').
                 sort_values(['rating'], ascending=False))
    print(user_full)
    #print('user {0} has already rated {1} movies.'. format(userID, user_full.shape[0]))
    #print('Recommending hightest {0} precdicted ratings movies not already rated.'.format(num_recommedations))
    
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             left_on='movieId', right_on='movieId').
                       rename(columns={user_row_number: 'Predictions'}).
                       sort_values('Predictions', ascending=False).
    iloc[:num_recommedations, :-1])
    return user_full, recommendations

#already_rated, predictions = recommend_movies(predicted_rating_df, 2, movies_df, ratings_df, 10)

#already_rated = already_rated.head(10)
#predictions = predictions

userID = 2 
num_recommedations = 10
original_ratings_df = ratings_df

user_row_number = userID-1
sorted_user_predictions = predicted_rating_df.iloc[user_row_number].sort_values(ascending=False)

user_data = original_ratings_df[original_ratings_df.userId == (userID)]
user_full = (user_data.merge(movies_df, how= 'left', left_on='movieId', right_on='movieId').
             sort_values(['rating'], ascending=False))

#print('user {0} has already rated {1} movies.'. format(userID, user_full.shape[0]))
#print('Recommending hightest {0} precdicted ratings movies not already rated.'.format(num_recommedations))
#
recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                   merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left'
                          ,left_on='movieId',right_on='movieId').
                 
                   rename(columns={user_row_number: 'Predictions'}).
                   sort_values('Predictions', ascending=False).
iloc[:num_recommedations, :-1])
#print(recommendations)
    