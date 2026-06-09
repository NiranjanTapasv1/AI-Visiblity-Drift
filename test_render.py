import streamlit as st
st.set_page_config(page_title="Test", layout="wide")
st.markdown("""
<style>
.stApp { background: #0a0b14 !important; }
h1 { color: white !important; }
p { color: #9294a8 !important; }
</style>
""", unsafe_allow_html=True)
st.markdown("<h1>App is alive</h1>", unsafe_allow_html=True)
st.write("If you see this, Streamlit renders correctly.")
