import streamlit as st

def app():
    st.write('This is the home page of this TV show recommendation app.')

    st.write('You can either select the `Simple` or `Advanced` mode from the Navigation selectbox for getting recommendations of TV shows that are available on Netflix.')
    
    st.write('\n')

    st.write('`Simple Mode`: You are recommended TV shows using preset features.')

    st.write('`Advanced Mode`: You can select the features by which you want to get recommendations.')
    
    st.write('\n')
             
    st.write('(`Enabling Dark Mode`: To do so, go to the Taskbar located at the top right of the screen, then Settings, then enable the Dark mode theme.)')
