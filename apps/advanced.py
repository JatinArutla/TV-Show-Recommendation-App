import streamlit as st

def app():
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

        st.write('Select the features you want to be recommended shows by')
        col1, col2 = st.beta_columns(2)
        with col1:
            check_1 = st.checkbox('Genre')
            check_2 = st.checkbox('Release Year')
            check_3 = st.checkbox('Duration')
        with col2:
            check_4 = st.checkbox('Plot/ Summary')
            check_5 = st.checkbox('Country')
            check_6 = st.checkbox('Cast')
        

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
        f = ['listed_in', 'release_year', 'duration', 'description', 'country', 'cast', 'rating']
        features = ['title', 'rating']

        # if check_1 is False and check_2 is False and check_3 is False and check_4 is False and check_5 is False and check_6 is False:
        #     for i in f:
        #         features.append(i)

        if check_1:
            features.append('listed_in')
        if check_2:
            features.append('release_year')
        if check_3:
            features.append('duration')
        if check_4:
            features.append('description')
        if check_5:
            features.append('country')
        if check_6:
            features.append('cast')
        
        df = df[features]
        for feature in features:
            df[feature] = df[feature].apply(str)
        for feature in features:
            df[feature] = df[feature].apply(cleaning_data)
        df['title'] = df['title'].apply(no_space)

        # def get_lemmatized_text(corpus):
        #     lemmatizer = WordNetLemmatizer()
        #     return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]
        # if check_4:
        #     df['description'] = get_lemmatized_text(df['description'])

        def create_soup(a):
            b = ''
            for i in range(len(features)):
                b = b + ' ' + a[i]
            return b
        df['soup'] = df.apply(create_soup, axis=1)

        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(df['soup'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        df.reset_index(drop=True, inplace=True)
        indices = pd.Series(df.index, index=df['title'])
        # st.write('Number of TV shows to be recommended:')
        num1 = st.slider('Number of TV shows to be recommended:', 1, 10, 10)

        def recommendation(titles, cosine_sim, num1):
            titles = titles.replace(' ','').lower()
            idx = indices[titles]
            similarity = list(enumerate(cosine_sim[idx]))
            similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
            similarity = similarity[1:num1+1]
            shows_indices = [i[0] for i in similarity]
            arr = netflix_shows['title'].iloc[shows_indices]
            df = arr.to_frame()
            df.index = range(1,len(df)+1)
            return df

        result = recommendation(options, cosine_sim, num1)
        st.write('Here are the recommended TV shows which are available on Netflix and are most similar to your selection.')
        st.write(result)
