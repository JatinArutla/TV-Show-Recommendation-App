import numpy as np
import pandas as pd
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer

netflix_overall = pd.read_csv("netflix_titles.csv")
netflix_overall.head()
netflix_shows = netflix_overall[netflix_overall['type']=='TV Show']

netflix_shows = netflix_shows.copy()
netflix_shows[['duration', 'season']] = netflix_shows['duration'].str.split(' ', 1, expand=True)
netflix_shows.drop(columns=['season'], axis=1, inplace=True)

netflix_shows[['genre', 'listed_in']] = netflix_shows['listed_in'].str.split(' ', 1, expand=True)
netflix_shows['rating'] = netflix_shows['rating'].replace('TV-MA', 'R')
netflix_shows['rating'] = netflix_shows['rating'].replace('TV-14', 'TV-PG')
netflix_shows['rating'] = netflix_shows['rating'].replace('TV-G', 'TV-PG')

df = netflix_shows.fillna('')
def cleaning_data(x):
        return str.lower(x.replace(",", " "))
def no_space(x):
        return str.lower(x.replace(" ", ""))
df['release_year'] = df['release_year'].apply(str)
features = ['title', 'cast', 'description', 'genre', 'duration', 'release_year', 'country', 'rating']
df = df[features]
for feature in features:
    df[feature] = df[feature].apply(str)
x = ['title']
for feature in features:
    df[feature] = df[feature].apply(cleaning_data)
for feature in x:
    df[feature] = df[feature].apply(no_space)
    
l1 = ['4', '1', '2', '3', '5', '9', '8', '6', '11', '7', '13', '12',
       '10', '16', '15']
l2 = ['four', 'one', 'two', 'three', 'five', 'nine', 'eight', 'six', 'eleven', 'seven', 'thirteen', 'twelve',
     'ten', 'sixteen', 'fifteen']
df['duration'] = df['duration'].replace(l1, l2)   
    
def get_lemmatized_text(corpus):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]
df['description'] = get_lemmatized_text(df['description'])

tokenizer=ToktokTokenizer()
stopword_list=nltk.corpus.stopwords.words('english')

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
df['description'] = df['description'].apply(remove_stopwords)

df.to_csv('preprocessed.csv')