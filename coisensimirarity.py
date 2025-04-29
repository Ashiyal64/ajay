
import streamlit as st
import pandas as pd
from io import BytesIO

from PIL import Image
import requests
import uuid

from streamlit_option_menu import option_menu

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("productdata.csv")
print(df)

with st.sidebar:
    selected=option_menu("",["Image"])




# Vectorize=TfidfVectorizer(stop_words='english')]
# X=Vectorize.fit_transform(df['Description'])
#
# user_input=input("Enter your name decrption")
#
# user_input_vector=Vectorize.transform([user_input])
# cosine_simiralit=cosine_similarity(user_input_vector,X).flatten()
# similar_indices=cosine_simiralit.argsort()[-3:][::-1]
# recommendation=df.iloc[similar_indices]
# print(recommendation)
#


if "shown" not in st.session_state:
    st.session_state.shown = None
if selected=="Image":

    if st.session_state.shown is None:
        st.subheader("Product Gallery")
        new_width, new_height = 300, 200
        cols = st.columns(4)

        # — Loop and display —
        for i, row in df.iterrows():
            c = cols[i % 4]
            url = row["image_url"]
            caption = row["name"]
            with c:
                # Fetch the image bytes
                resp = requests.get(url, timeout=5)

                if resp.status_code == 200:
                    # Open with Pillow
                    img = Image.open(BytesIO(resp.content))

                    # Resize exactly to (new_width, new_height)
                    img = img.resize((new_width, new_height), resample=Image.LANCZOS)

                    # Display
                    st.image(img, caption=caption, use_container_width=False)
                else:
                    # Fallback placeholder on any HTTP error
                    placeholder = (
                        f"https://via.placeholder.com/{new_width}x{new_height}"
                        f"?text=No+Image"
                    )
                    st.image(
                        placeholder,
                        caption=f"{caption} (HTTP {resp.status_code})",
                        use_container_width = False
                    )
                if st.button(df['name'][i], key=f"btn_{i}"):
                    st.session_state.shown = i
                    st.rerun()
    else:
        clicked_index = st.session_state.shown
        st.subheader(f"Similarity for {df['Description'][clicked_index]}")

        # TF-IDF similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['name'])
        user_vector = vectorizer.transform([df["name"][clicked_index]])
        cosine_sim = cosine_similarity(user_vector, X).flatten()
        similar_indices = cosine_sim.argsort()[::-1]
        similar_indices = [idx for idx in similar_indices if idx != clicked_index][:3]  # Top 3
        rec_cols = st.columns(3)
        for idx, col in zip(similar_indices, rec_cols):
            with col:
                url = df["image_url"][idx]
                caption = df["name"][idx]

                # 1) Fetch
                resp = requests.get(url, timeout=5)

                if resp.status_code == 200:
                    new_width, new_height = 300, 200
                    # 2) Open & resize via Pillow
                    img = Image.open(BytesIO(resp.content))
                    img = img.resize((new_width, new_height), resample=Image.LANCZOS)

                    # 3) Display with use_container_width
                    st.image(
                        img,
                        caption=caption,
                        use_container_width=False
                    )
                else:
                    # fallback placeholder
                    placeholder = (
                        f"https://via.placeholder.com/{new_width}x{new_height}"
                        f"?text=No+Image"
                    )
                    st.image(
                        placeholder,
                        caption=f"{caption} (HTTP {resp.status_code})",
                        use_container_width=False
                    )

                # your existing captions
                st.caption(f"**Description:** {df['Description'][idx]}")
                st.caption(f"**Category:**    {df['Category'][idx]}")

        # Back button to return to gallery
        if st.button("🔙 Back to Gallery"):
            st.session_state.shown = None
            st.rerun()











