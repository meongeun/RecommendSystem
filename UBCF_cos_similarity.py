# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from  sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/rating.csv', sep=';')
movies = pd.read_csv('C:/Users/user/Desktop/KME/codeing/backend/data/movies.csv', sep=';', encoding='latin-1')

merged = ratings.merge(movies, left_on= 'movieId', right_on='movieId', sort=True)

merged = merged[['userId', 'title', 'rating']]

movieRatings = merged.pivot_table(index=['userId'], columns=['title'], values = 'rating')

#print(type(movieRatings))
#movieRatings.replace({np.nan:0}, regex=True, inplace=True)
movieRatings.fillna(0, inplace=True)

user_similarity = cosine_similarity(movieRatings)

user_sim_df = pd.DataFrame(user_similarity, index=movieRatings.index, columns = movieRatings.index)

movieRatings = movieRatings.T

import operator

def recommendation(user):
    if user not in movieRatings.columns:
        return('No data available on this User')
    
    sim_user = user_sim_df.sort_values(by=user, ascending=False).index[1:11]
    
    best = []
    
    for i in sim_user:
        max_score = movieRatings.loc[:,i].max()
        best.append(movieRatings[movieRatings.loc[:,i] == max_score].index.tolist())
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
    return sorted_list

print(recommendation(2))
            