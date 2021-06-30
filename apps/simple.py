import streamlit as st
import numpy as np
import pandas as pd

def app():
    import streamlit as st
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
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

        df = pd.read_csv('preprocessed.csv')

        features1 = ['title', 'cast', 'description', 'genre', 'duration', 'release_year', 'country', 'rating']
        df = df[features1]
        for feature in features1:
            df[feature] = df[feature].apply(str)

        def create_soup(a):
            return a['title'] + ' ' + a['description'] + ' ' + a['rating']
        df['soup'] = df.apply(create_soup, axis=1)
        count = TfidfVectorizer(stop_words='english')
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

        result= recommendation(options, cosine_sim)
        st.write('Here are the recommended TV shows which are available on Netflix and are most similar to your selection.')
        st.write(result)
