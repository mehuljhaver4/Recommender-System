import pandas as pd
import numpy as np
from ast import literal_eval
import warnings; warnings.simplefilter('ignore')


meta = pd.read_csv('Data/movies_metadata.csv')
meta['genres'] = meta['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
meta['year'] = pd.to_datetime(meta['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


genres = meta.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
genres.name = 'genre'
gen_meta = meta.drop('genres', axis=1).join(genres)


def weighted_rating(df,min_votes,avg_rating):
    votes = df['vote_count']
    rating = df['vote_average']
    return (votes/(votes+min_votes)*rating) + (min_votes/(min_votes+votes)*avg_rating)

def top_movies(genre, percentile = 0.80):

    df = gen_meta[gen_meta['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    avg_rating = round(vote_averages.mean(),3)
    min_votes = vote_counts.quantile(percentile)

    qualified_movies = df[(df['vote_count'] >= min_votes) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())& (df['vote_average']>= avg_rating)][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genre']]
    qualified_movies['vote_count'] = qualified_movies['vote_count'].astype('int')
    qualified_movies['vote_average'] = qualified_movies['vote_average'].astype('int')
    
    qualified_movies['Weighted_Rating'] = qualified_movies.apply(weighted_rating,args=(min_votes,avg_rating),axis = 1)
    qualified_movies = qualified_movies.sort_values('Weighted_Rating', ascending= False)

    print("Top 20",genre,"movies are:\n")
    print(qualified_movies.head(20)) 

top_movies('Romance')