"""Simple Streamlit UI for the book recommendation system."""

from pathlib import Path

import requests
import streamlit as st

from book_recommender import build_recommender, recommend_books


DATASET_PATH = Path(__file__).with_name("anotherOne.csv")


@st.cache_resource
def load_artifacts():
    return build_recommender(str(DATASET_PATH))


@st.cache_data(show_spinner=False)
def fetch_book_cover(title: str, author: str) -> str | None:
    """Fetch a book cover URL from the Google Books API."""
    query = f"intitle:{title} inauthor:{author}".strip()
    url = "https://www.googleapis.com/books/v1/volumes"

    try:
        response = requests.get(url, params={"q": query, "maxResults": 1}, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return None

    items = data.get("items", [])
    if not items:
        return None

    image_links = items[0].get("volumeInfo", {}).get("imageLinks", {})
    return (
        image_links.get("thumbnail")
        or image_links.get("smallThumbnail")
    )


def show_book_card(column, row, button_key: str) -> None:
    """Render one recommendation inside a Streamlit column."""
    cover_url = fetch_book_cover(row["Title"], row["Author"])

    with column:
        if cover_url:
            st.image(cover_url, use_container_width=True)
        else:
            st.caption("Image not available")

        st.markdown(f"**{row['Title']}**")
        st.write(f"Author: {row['Author']}")

        if st.button("View Details", key=button_key, use_container_width=True):
            st.session_state["selected_book"] = row["Title"]


def show_book_details(results, selected_title: str) -> None:
    """Display extra details for the selected recommendation."""
    selected_rows = results[results["Title"] == selected_title]
    if selected_rows.empty:
        return

    row = selected_rows.iloc[0]
    st.subheader(f"Details: {row['Title']}")
    st.write(f"Genre: {row['Genre']}")
    st.write(f"Average Rating: {row['Rating Score']}")
    st.write("Top 3 Reviews:")

    top_reviews = row["Top Reviews"]
    if top_reviews:
        for index, review in enumerate(top_reviews, start=1):
            st.write(f"{index}. {review}")
    else:
        st.write("No reviews available for this book.")


st.set_page_config(page_title="Book Recommendation System", layout="wide")
st.title("Book Recommendation System")
st.write(
    "Search by book title, author name, genre, or any keywords from the book content."
)

artifacts = load_artifacts()
query = st.text_input("Enter your query")

if st.button("Get Recommendations") or query:
    results = recommend_books(query, artifacts, top_n=5)

    if "message" in results.columns:
        st.info(results.iloc[0]["message"])
    else:
        st.subheader("Top Recommended Books")
        columns = st.columns(5)

        for index, (_, row) in enumerate(results.iterrows()):
            show_book_card(columns[index], row, button_key=f"view_details_{row['Title']}_{index}")

        selected_title = st.session_state.get("selected_book")
        if selected_title:
            st.divider()
            show_book_details(results, selected_title)
