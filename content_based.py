import pandas as pd
import numpy as np
import math
Ratings=pd.read_csv("C:/Users/tomar/OneDrive/Desktop/Recommender system/ratings.csv",encoding="ISO-8859-1")
Movies=pd.read_csv("C:/Users/tomar/OneDrive/Desktop/Recommender system/movies.csv",encoding="ISO-8859-1")
Tags=pd.read_csv("C:/Users/tomar/OneDrive/Desktop/Recommender system/tags.csv",encoding="ISO-8859-1")

a=Ratings[Ratings['userId']==320]
a[a['movieId']==260]
TF= Tags.groupby(['movieId','tag'], as_index = False, sort = False).count().rename(columns = {'userId': 'tag_count_TF'})[['movieId','tag','tag_count_TF']]

Tag_distinct = Tags[['tag','movieId']].drop_duplicates()
DF =Tag_distinct.groupby(['tag'], as_index = False, sort = False).count().rename(columns = {'movieId': 'tag_count_DF'})[['tag','tag_count_DF']]
a=math.log10(len(np.unique(Tags['movieId'])))
DF['IDF']=a-np.log10(DF['tag_count_DF'])
TF = pd.merge(TF,DF,on = 'tag', how = 'left', sort = False)
TF['TF-IDF']=TF['tag_count_TF']*TF['IDF']

Vect_len=TF[['movieId','TF-IDF']]
Vect_len['TF-IDF-Sq']=Vect_len['TF-IDF']**2
Vect_len =Vect_len.groupby(['movieId'], as_index = False, sort = False).sum().rename(columns = {'TF-IDF-Sq': 'TF-IDF-Sq-sum'})[['movieId','TF-IDF-Sq-sum']]
Vect_len['vect_len'] = np.sqrt(Vect_len[['TF-IDF-Sq-sum']].sum(axis=1))
TF = pd.merge(TF,Vect_len,on = 'movieId', how = 'left', sort = False)
TF['TAG_WT']=TF['TF-IDF']/TF['vect_len']
Ratings_filter=Ratings[Ratings['rating']>=3.5]
distinct_users=np.unique(Ratings['userId'])
user_tag_pref=pd.DataFrame()
i=1

#enter user ID for analysis
userID = 320

user_index = distinct_users.tolist().index(userID)

for user in distinct_users[user_index:user_index+1]:
    
    if i%30==0:
        print ("user: ", i , "out of: ", len(distinct_users))
            
    user_data=  Ratings_filter[Ratings_filter['userId']==user]
    user_data = pd.merge(TF,user_data,on = 'movieId', how = 'inner', sort = False)
    user_data1 = user_data.groupby(['tag'], as_index = False, sort = False).sum().rename(columns = {'TAG_WT': 'tag_pref'})[['tag','tag_pref']]
    user_data1['user']=user
    user_tag_pref = user_tag_pref.append(user_data1, ignore_index=True)
    i=i+1
    distinct_users=np.unique(Ratings_filter['userId'])
tag_merge_all=pd.DataFrame()

i=1

#enter user ID for analysis
userID = 320

user_index = distinct_users.tolist().index(userID)

for user in distinct_users[user_index:user_index+1]:

    
    user_tag_pref_all=  user_tag_pref[user_tag_pref['user']==user]
    distinct_movies = np.unique(TF['movieId'])
    j=1
    for movie in distinct_movies:
        
        if j%300==0:
            
            print ("movie: ", j , "out of: ", len(distinct_movies) , "with user: ", i , "out of: ", len(distinct_users))
        
        TF_Movie=  TF[TF['movieId']==movie]
        tag_merge = pd.merge(TF_Movie,user_tag_pref_all,on = 'tag', how = 'left', sort = False)
        tag_merge['tag_pref']=tag_merge['tag_pref'].fillna(0)
        tag_merge['tag_value']=tag_merge['TAG_WT']*tag_merge['tag_pref']
        
        TAG_WT_val=np.sqrt(np.sum(np.square(tag_merge['TAG_WT']), axis=0))
        tag_pref_val=np.sqrt(np.sum(np.square(user_tag_pref_all['tag_pref']), axis=0))
        
        tag_merge_final = tag_merge.groupby(['user','movieId'])[['tag_value']].sum().rename(columns = {'tag_value': 'Rating'}).reset_index()
        
        tag_merge_final['Rating']=tag_merge_final['Rating']/(TAG_WT_val*tag_pref_val)
        
        tag_merge_all = tag_merge_all.append(tag_merge_final, ignore_index=True)
        j=j+1
    i=i+1
tag_merge_all = tag_merge_all.sort_values(['user','Rating'], ascending=False)
# Removing movies already rated by user

movies_rated = Ratings[Ratings['userId']==userID]['movieId']
tag_merge_all = tag_merge_all[~tag_merge_all['movieId'].isin(movies_rated)]

tag_merge_all
distinct_users=np.unique(Ratings['userId'])
user_tag_pref=pd.DataFrame()
i=1

#enter user ID for analysis
userID = 320

user_index = distinct_users.tolist().index(userID)

for user in distinct_users[user_index:user_index+1]:    
    if i%30==0:
        print ("user: ", i , "out of: ", len(distinct_users))
            
    user_data=  Ratings[Ratings['userId']==user]
    user_data['weight']=user_data["rating"]-user_data["rating"].mean()
    user_data1 = pd.merge(TF,user_data,on = 'movieId', how = 'inner', sort = False)
    user_data1['TAG_WT_WTD'] = user_data1['TAG_WT']*user_data1['weight']
    user_data2 = user_data1.groupby(['tag'], as_index = False, sort = False).sum().rename(columns = {'TAG_WT_WTD': 'tag_pref'})[['tag','tag_pref']]
    user_data2['user']=user
    user_tag_pref = user_tag_pref.append(user_data2, ignore_index=True)
    i=i+1
    
    distinct_users=np.unique(Ratings_filter['userId'])
tag_merge_all=pd.DataFrame()

i=1

#enter user ID for analysis
userID = 320

user_index = distinct_users.tolist().index(userID)

for user in distinct_users[user_index:user_index+1]:

    
    user_tag_pref_all=  user_tag_pref[user_tag_pref['user']==user]
    distinct_movies = np.unique(TF['movieId'])
    j=1
    for movie in distinct_movies:
        
        if j%300==0:
            
            print ("movie: ", j , "out of: ", len(distinct_movies) , "with user: ", i , "out of: ", len(distinct_users))
        
        TF_Movie=  TF[TF['movieId']==movie]
        tag_merge = pd.merge(TF_Movie,user_tag_pref_all,on = 'tag', how = 'left', sort = False)
        tag_merge['tag_pref']=tag_merge['tag_pref'].fillna(0)
        tag_merge['tag_value']=tag_merge['TAG_WT']*tag_merge['tag_pref']
        
        TAG_WT_val=np.sqrt(np.sum(np.square(tag_merge['TAG_WT']), axis=0))
        tag_pref_val=np.sqrt(np.sum(np.square(user_tag_pref_all['tag_pref']), axis=0))
         tag_merge_final = tag_merge.groupby(['user','movieId'])[['tag_value']].sum().rename(columns = {'tag_value': 'Rating'}).reset_index()
        
        tag_merge_final['Rating']=tag_merge_final['Rating']/(TAG_WT_val*tag_pref_val)
        
        tag_merge_all = tag_merge_all.append(tag_merge_final, ignore_index=True)
        j=j+1
    i=i+1
tag_merge_all = tag_merge_all.sort_values(['user','Rating'], ascending=False)

# Removing movies already rated by user

movies_rated = Ratings[Ratings['userId']==userID]['movieId']
tag_merge_all = tag_merge_all[~tag_merge_all['movieId'].isin(movies_rated)]
tag_merge_all
