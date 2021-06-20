import streamlit as st
import numpy as np
import pandas as pd

def app():
    import streamlit as st
    import pandas as pd
    import numpy as np
    # import nltk
    # from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    header = st.beta_container()
    pre_processing = st.beta_container()

    with header:

        netflix_data = pd.read_csv("netflix_titles.csv")
        netflix_shows = netflix_data[netflix_data['type']=='TV Show']
        netflix_shows = netflix_shows.copy()
        netflix_shows['title'] = netflix_shows['title'].apply(str)
        arr = netflix_shows['title']
        arr = arr.tolist()
        sel_col, disp_col = st.beta_columns(2)
        options = sel_col.selectbox('Select a TV show (Just overwrite the title in the search box)', options=arr)

    with pre_processing:

        netflix_shows[['duration', 'season']] = netflix_shows['duration'].str.split(' ', 1, expand=True)
        netflix_shows.drop(columns=['season'], axis=1, inplace=True)
        netflix_shows['rating'] = netflix_shows['rating'].replace('TV-MA', 'R')
        netflix_shows['rating'] = netflix_shows['rating'].replace('TV-14', 'TV-PG')
        netflix_shows['rating'] = netflix_shows['rating'].replace('TV-G', 'TV-PG')

        def cleaning_data(a):
            return str.lower(a.replace(",", " "))
        def no_space(x):
            return str.lower(x.replace(" ", ""))

        df = netflix_shows.fillna('')
        df.head(2)
        df['release_year'] = df['release_year'].apply(str)
        features = ['title', 'description', 'rating']
        df = df[features]
        for feature in features:
            df[feature] = df[feature].apply(str)
        for feature in features:
            df[feature] = df[feature].apply(cleaning_data)
        df['title'] = df['title'].apply(no_space)

        
        # def get_lemmatized_text(corpus):
        #     lemmatizer = WordNetLemmatizer()
        #     return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

        # df['description'] = get_lemmatized_text(df['description'])

        def create_soup(a):
            return a['title'] + ' ' + a['description'] + ' ' + a['rating']
        df['soup'] = df.apply(create_soup, axis=1)

        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(df['soup'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        df.reset_index(drop=True, inplace=True)
        indices = pd.Series(df.index, index=df['title'])

        def recommendation(titles, cosine_sim):
            titles = titles.replace(' ','').lower()
            idx = indices[titles]
            similarity = list(enumerate(cosine_sim[idx]))
            similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
            similarity = similarity[1:11]
            shows_indices = [i[0] for i in similarity]
            arr = netflix_shows['title'].iloc[shows_indices]
            df = arr.to_frame()
            df.index = range(1,len(df)+1)
            return df

        result = recommendation(options, cosine_sim)
        st.write('Here are the recommended TV shows which are available on Netflix and are most similar to your selection.')
        st.write(result)
