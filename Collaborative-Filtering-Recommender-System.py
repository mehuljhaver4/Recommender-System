# Importing the required libraries

import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

import warnings; warnings.simplefilter('ignore')


reader = Reader()
ratings = pd.read_csv('Data/ratings_small.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

svd = SVD()
cross_validate(svd, data, measures=['RMSE','MAE'], cv=5)

trainset = data.build_full_trainset()
svd.fit(trainset)

svd.predict(1, 302, 3)


