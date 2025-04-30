import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image
import requests
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your product data
df = pd.read_csv("productdata.csv")

# Sidebar navigation
with st.sidebar:
    selected = option_menu("", ["Image"])

# Custom CSS styling
st.markdown("""
    <style>
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 10px;
    }
    .element-container button {
        background-color: #f0f2f6;
        border: none;
        color: #333;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-top: 5px;
        transition: all 0.2s ease-in-out;
    }
    .element-container button:hover {
        background-color: #cce5ff;
        color: black;
        transform: scale(1.05);
    }
    .caption-text {
        font-size: 14px;
        color: #555;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state for selection
if "shown" not in st.session_state:
    st.session_state.shown = None

if selected == "Image":
    if st.session_state.shown is None:
        st.subheader("🖼️ Product Gallery")
        new_width, new_height = 300, 200
        cols = st.columns(4)

        for i, row in df.iterrows():
            c = cols[i % 4]
            url = row["image_url"]
            caption = row["name"]
            with c:
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        img = Image.open(BytesIO(resp.content))
                        img = img.resize((new_width, new_height), resample=Image.LANCZOS)
                        st.image(img, caption=caption, use_container_width=False)
                    else:
                        placeholder = f"https://via.placeholder.com/{new_width}x{new_height}?text=No+Image"
                        st.image(placeholder, caption=f"{caption} (HTTP {resp.status_code})", use_container_width=False)
                except:
                    placeholder = f"https://via.placeholder.com/{new_width}x{new_height}?text=No+Image"
                    st.image(placeholder, caption="Error fetching image", use_container_width=False)

                if st.button(df['name'][i], key=f"btn_{i}"):
                    st.session_state.shown = i
                    st.rerun()

    else:
        clicked_index = st.session_state.shown
        st.subheader(f"🔍 Similar Products to: *{df['name'][clicked_index]}*")

        # TF-IDF similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['name'])
        user_vector = vectorizer.transform([df["name"][clicked_index]])
        cosine_sim = cosine_similarity(user_vector, X).flatten()
        similar_indices = cosine_sim.argsort()[::-1]
        similar_indices = [idx for idx in similar_indices if idx != clicked_index][:3]

        rec_cols = st.columns(3)
        for idx, col in zip(similar_indices, rec_cols):
            with col:
                url = df["image_url"][idx]
                caption = df["name"][idx]
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        img = Image.open(BytesIO(resp.content))
                        img = img.resize((new_width, new_height), resample=Image.LANCZOS)
                        st.image(img, caption=caption, use_container_width=False)
                    else:
                        placeholder = f"https://via.placeholder.com/{new_width}x{new_height}?text=No+Image"
                        st.image(placeholder, caption=f"{caption} (HTTP {resp.status_code})", use_container_width=False)
                except:
                    placeholder = f"https://via.placeholder.com/{new_width}x{new_height}?text=No+Image"
                    st.image(placeholder, caption="Image error", use_container_width=False)

                st.markdown(f"<div class='caption-text'><strong>Description:</strong> {df['Description'][idx]}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='caption-text'><strong>Category:</strong> {df['Category'][idx]}</div>", unsafe_allow_html=True)

        # Back to gallery
        if st.button("🔙 Back to Gallery"):
            st.session_state.shown = None
            st.rerun()
