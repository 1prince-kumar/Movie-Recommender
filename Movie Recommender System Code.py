import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer

#reading datasets
df_credits= pd.read_csv('credits.csv')
df_keywords= pd.read_csv('keywords.csv')
df_links= pd.read_csv('links_small.csv')
df_md= pd.read_csv('movies_metadata.csv')
#df_ratings_small= pd.read_csv('ratings_small.csv')    
    
#cleaning datasets
df_md=df_md.drop(['original_title','spoken_languages','imdb_id','adult','belongs_to_collection','original_title','budget','homepage','poster_path','revenue','production_countries','runtime','status','video'],axis=1)
df_md['genres']=df_md['genres'].fillna('[]').apply(literal_eval).apply(lambda x:[i['name'] for i in x] if isinstance(x, list) else [])
df_md['production_companies']=df_md['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x:[i['name'] for i in x] if isinstance(x, list) else [])
df_links=df_links[df_links['tmdbId'].notnull()]['tmdbId'].astype('int')

#drop rows with incorrect format of id
df_md=df_md.drop([19730,29503,35587])
df_md['id']=df_md['id'].astype(int)
#to get the above indices
df_md=df_md[df_md['id']!='2014-01-01']

#merging keywords and credits information
df_keywords['id']=df_keywords['id'].astype('int')
df_credits['id']=df_credits['id'].astype('int')
df_md=df_md.merge(df_credits,on='id')
df_md=df_md.merge(df_keywords,on='id')

#getting a subset of df_md
df_md=df_md[df_md['id'].isin(df_links)]

df_md['overview']=df_md['overview'].fillna('')
df_md['tagline']=df_md['tagline'].fillna('')
#df_md['description']=df_md['overview']+df_md['tagline']
#df_md['description']=df_md['description'].fillna('')
df_md['cast']=df_md['cast'].apply(literal_eval)
df_md['crew']=df_md['crew'].apply(literal_eval)
df_md['keywords']=df_md['keywords'].apply(literal_eval)
    
#extracting director
def get_director(x):
    for i in x:
        if i['job']=='Director':
            return i['name']
    return np.nan

df_md['Director']=df_md['crew'].apply(get_director)
df_md['Director']=df_md['Director'].fillna('')

#getting lead roles
def get_leadroles(x):
    names=[]
    for i in x:
        try:
            names.append(i['name'])
        except:
            t=0
    if len(x)>=4:
        return names[:4]
    else:
        return names
        
df_md['lead_roles']=df_md['cast'].apply(get_leadroles)

#getting keywords
def get_keywords(x):
    names=[]
    for i in x:
        try:
            names.append(i['name'])
        except:
            t=0
    return names

df_md['keywords']=df_md['keywords'].apply(get_keywords)

#preprocessing production companies and overview
stemmer=SnowballStemmer('english')
df_md['production_companies']=df_md['production_companies'].apply(lambda x: [str.lower(i.replace(' ','')) for i in x])
df_md['overview']=df_md['overview'].apply(lambda x: stemmer.stem(x))
df_md['overview']=df_md['overview'].apply(lambda x: str.lower(x.replace(' ','')))

#cleaning columns
df_md['lead_roles']=df_md['lead_roles'].apply(lambda x: [str.lower(i.replace(' ','')) for i in x])
df_md['Director']=df_md['Director'].apply(lambda x: str.lower(x.replace(' ','')))
df_md['Director']=df_md['Director'].apply(lambda x: [x,x,x])
#preprocessing keywords
df_md['keywords']=df_md['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
df_md['keywords']=df_md['keywords'].apply(lambda x: [str.lower(i.replace(' ','')) for i in x])

df_md['combined']=df_md['keywords']+df_md['Director']+df_md['lead_roles']+df_md['genres']+df_md['production_companies']
df_md['combined']=df_md['combined'].apply(lambda x: ' '.join(x))
df_md['combined']=df_md['combined']+df_md['overview']

#making a sparse matrix
cv=CountVectorizer(analyzer='word',ngram_range=(1,2),min_df=0,stop_words='english')
cv_matrix=cv.fit_transform(df_md['combined'])

cosine_sim=cosine_similarity(cv_matrix,cv_matrix)

df_md=df_md.reset_index()
titles=df_md['title'].apply(lambda x: x.lower())
indices=pd.Series(df_md.index, index=titles)

#save df_md,cosine_sim
df_md=df_md.drop(['index','overview','genres','id','original_language','overview','popularity','production_companies','release_date','tagline','cast','crew','keywords','Director','lead_roles'],axis=1)
#df_md.to_csv('movie_data.csv',index=False)
#np.save('cosine_matrix',cosine_sim)
#improved recommendations using Rating method as used by IMDB
vote_counts=df_md[df_md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages=df_md[df_md['vote_average'].notnull()]['vote_average'].astype('int')
c=vote_averages.mean()
m=vote_counts.quantile(0.95)
def weighted_rating(x):
    v=x['vote_count']
    r=x['vote_average']
    return (v/(v+m)*r)+(m/(m+v)*c)

def improved_recommendations(title):
    if title in titles.unique():
        title=title.lower()
        y=indices[title]
        similarity=list(enumerate(cosine_sim[y]))
        similarity=sorted(similarity,key=lambda x: x[1],reverse=True)
        similarity=similarity[1:31]
        movie_indices=[i[0] for i in similarity]
        movies=df_md.iloc[movie_indices][['title','vote_count','vote_average']]
        best_movies=movies
        best_movies['vote_count']=best_movies['vote_count'].astype('int')
        best_movies['vote_average']=best_movies['vote_average'].astype('int')
        best_movies['weighted_rating']=best_movies.apply(weighted_rating,axis=1)
        best_movies=best_movies.sort_values('weighted_rating', ascending= False).head(10)
        best_movies=best_movies.reset_index()
        return best_movies['title']
    else:
        print('Please enter correct movie name or enter another movie name')


improved_recommendations('mean girls')

    
    











