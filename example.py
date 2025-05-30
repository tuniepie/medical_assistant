import streamlit as st
import pandas as pd


st.title("A Simple Streamlit Web App")
user_name = st.text_input("Please enter your name", '')
st.write(f"Hello {user_name}!")
number1 = st.slider("Choose an integer for x", 0, 10, 1)
number2 = st.slider("Choose an integer for y", 0, 10, 1)
data_frame = pd.DataFrame({"x": [number1], "y": [number2], "x + y": [number1 + number2]}, index = ["addition row"])
st.write(data_frame)