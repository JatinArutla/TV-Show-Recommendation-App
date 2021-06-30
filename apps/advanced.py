import streamlit as st

def app():
    import pandas as pd
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

        df = pd.read_csv('preprocessed.csv')
        features = ['title', 'rating']

        if check_1:
            features.append('genre')
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

        result= recommendation(options, cosine_sim, num1)
        st.write('Here are the recommended TV shows which are available on Netflix and are most similar to your selection.')
        st.write(result)
