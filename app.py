import streamlit as st
from multiapp import MultiApp
from apps import home, simple, advanced

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

darkmode = """
    <style>
    body {
    color: white;
    }
    h1 {
    color: white;
    }
    </style>
    """
st.markdown(darkmode, unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: #0E1117;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

app = MultiApp()

app.add_app("Home", home.app)
app.add_app("Simple", simple.app)
app.add_app("Advanced", advanced.app)

app.run()
