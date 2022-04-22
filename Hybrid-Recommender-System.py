# Importing the required libraries

import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

import warnings; warnings.simplefilter('ignore')

links_small = pd.read_csv('Data/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

meta = pd.read_csv('Data/movies_metadata.csv')
meta['genres'] = meta['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
meta['year'] = pd.to_datetime(meta['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

meta = meta.drop([19730, 29503, 35587]) #Incorrect data so remove it.
meta['id'] = meta['id'].astype('int')

sub_meta = meta[meta['id'].isin(links_small)]

sub_meta['tagline'] = sub_meta['tagline'].fillna('')
sub_meta['description'] = sub_meta['overview'] + sub_meta['tagline']
sub_meta['description'] = sub_meta['description'].fillna('')


credits = pd.read_csv('Data/credits.csv')
keywords = pd.read_csv('Data/keywords.csv')

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
meta['id'] = meta['id'].astype('int')

meta = meta.merge(credits, on='id')
meta = meta.merge(keywords, on='id')

sub_meta = meta[meta['id'].isin(links_small)]

sub_meta['cast'] = sub_meta['cast'].apply(literal_eval)
sub_meta['crew'] = sub_meta['crew'].apply(literal_eval)
sub_meta['keywords'] = sub_meta['keywords'].apply(literal_eval)
sub_meta['cast_size'] = sub_meta['cast'].apply(lambda x: len(x))
sub_meta['crew_size'] = sub_meta['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

sub_meta['director'] = sub_meta['crew'].apply(get_director)

sub_meta['cast'] = sub_meta['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
sub_meta['cast'] = sub_meta['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

sub_meta['keywords'] = sub_meta['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

sub_meta['cast'] = sub_meta['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

sub_meta['director'] = sub_meta['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
sub_meta['director'] = sub_meta['director'].apply(lambda x: [x,x, x])

# Preprocessing of keywords

key = sub_meta.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
key.name = 'keyword'
key = key.value_counts()

# Only taking keywords into considerations whose count is more than 1.
key = key[key > 1]

def filter_keywords(x):
    words = []
    for i in x:
        if i in key:
            words.append(i)
    return words

stemmer = SnowballStemmer('english')

sub_meta['keywords'] = sub_meta['keywords'].apply(filter_keywords)
sub_meta['keywords'] = sub_meta['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
sub_meta['keywords'] = sub_meta['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

sub_meta['soup'] = sub_meta['keywords'] + sub_meta['cast'] + sub_meta['director'] + sub_meta['genres']
sub_meta['soup'] = sub_meta['soup'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(sub_meta['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

sub_meta = sub_meta.reset_index()
titles = sub_meta['title']
indices = pd.Series(sub_meta.index, index=sub_meta['title'])

def weighted_rating(df,min_votes,avg_rating):
    votes = df['vote_count']
    rating = df['vote_average']
    return (votes/(votes+min_votes)*rating) + (min_votes/(min_votes+votes)*avg_rating)


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
    

id_map = pd.read_csv('Data/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(sub_meta[['title', 'id']], on='id').set_index('title')

indices_map = id_map.set_index('id')

reader = Reader()
ratings = pd.read_csv('Data/ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE','MAE'], cv=5)

trainset = data.build_full_trainset()
svd.fit(trainset)

def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = sub_meta.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    print("Top 10 movies that user -",userId," may also like:\n")
    print(movies.head(10))

hybrid(1, 'Avatar')
