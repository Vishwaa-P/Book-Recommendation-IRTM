"""Beginner-friendly IR-based book recommendation helpers."""

from __future__ import annotations

import ast
import re
import string
from dataclasses import dataclass
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


NLTK_RESOURCES = ("stopwords", "punkt", "wordnet", "omw-1.4")
NLTK_RESOURCE_PATHS = {
    "stopwords": ("corpora/stopwords", "corpora/stopwords.zip"),
    "punkt": ("tokenizers/punkt", "tokenizers/punkt.zip"),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    "omw-1.4": ("corpora/omw-1.4", "corpora/omw-1.4.zip"),
}


@dataclass
class RecommenderArtifacts:
    books_df: pd.DataFrame
    tfidf_matrix: object
    vectorizer: TfidfVectorizer
    similarity_matrix: np.ndarray


def download_nltk_resources() -> None:
    """Download the NLTK datasets needed for preprocessing."""
    for resource in NLTK_RESOURCES:
        resource_found = False
        for resource_path in NLTK_RESOURCE_PATHS[resource]:
            try:
                nltk.data.find(resource_path)
                resource_found = True
                break
            except LookupError:
                continue
        if not resource_found:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def _get_stop_words() -> set[str]:
    return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def _get_lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def parse_authors(value: str) -> str:
    """Convert author lists like "['A', 'B']" into readable text."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return ", ".join(str(item).strip() for item in parsed if str(item).strip())
    except (ValueError, SyntaxError):
        pass
    return value


def preprocess_text(text: str) -> str:
    """Lowercase, clean, tokenize, remove stopwords, and lemmatize text."""
    lemmatizer = _get_lemmatizer()
    stop_words = _get_stop_words()

    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalnum() and token not in stop_words
    ]
    return " ".join(cleaned_tokens)


def _first_non_empty(series: pd.Series) -> str:
    for value in series:
        text = str(value).strip()
        if text:
            return text
    return ""


def _prepare_reviews(group: pd.DataFrame) -> list[str]:
    top_reviews = (
        group.sort_values(["review/score", "_row_order"], ascending=[False, True])
        .head(3)["review/text"]
        .tolist()
    )
    return [str(review).strip() for review in top_reviews if str(review).strip()]


def build_recommender(
    csv_path: str,
    nrows: int = 50000,
    max_features: int = 10000,
) -> RecommenderArtifacts:
    """Load data, clean it, aggregate reviews, and build TF-IDF artifacts."""
    download_nltk_resources()

    df = pd.read_csv(csv_path, nrows=nrows)
    df = df.fillna("")
    df["_row_order"] = np.arange(len(df))
    df["authors"] = df["authors"].apply(parse_authors)
    df["review/score"] = pd.to_numeric(df["review/score"], errors="coerce").fillna(0.0)

    grouped_rows = []
    for title, group in df.groupby("Title", sort=False):
        review_summary = " ".join(
            part.strip() for part in group["review/summary"].astype(str) if part.strip()
        )
        review_text = " ".join(
            part.strip() for part in group["review/text"].astype(str) if part.strip()
        )

        grouped_rows.append(
            {
                "Title": str(title).strip(),
                "authors": _first_non_empty(group["authors"]),
                "publisher": _first_non_empty(group["publisher"]),
                "description": _first_non_empty(group["description"]),
                "categories": _first_non_empty(group["categories"]),
                "review/summary": review_summary,
                "review/text": review_text,
                "rating_score": round(group["review/score"].mean(), 2),
                "top_reviews": _prepare_reviews(group),
            }
        )

    books_df = pd.DataFrame(grouped_rows)
    books_df["content"] = (
        books_df["Title"]
        + " "
        + books_df["authors"]
        + " "
        + books_df["review/summary"]
        + " "
        + books_df["review/text"]
        + " "
        + books_df["description"]
        + " "
        + books_df["categories"]
    )
    books_df["processed_content"] = books_df["content"].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(books_df["processed_content"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return RecommenderArtifacts(
        books_df=books_df,
        tfidf_matrix=tfidf_matrix,
        vectorizer=vectorizer,
        similarity_matrix=similarity_matrix,
    )


def _match_metadata(books_df: pd.DataFrame, query: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    query = query.lower().strip()
    title_matches = books_df["Title"].str.lower().str.contains(query, na=False)
    author_matches = books_df["authors"].str.lower().str.contains(query, na=False)
    category_matches = books_df["categories"].str.lower().str.contains(query, na=False)
    return title_matches, author_matches, category_matches


def recommend_books(
    query: str,
    artifacts: RecommenderArtifacts,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend books by title/author/genre lookup or keyword search."""
    if not isinstance(query, str) or not query.strip():
        return pd.DataFrame({"message": ["Please enter a book title, author, genre, or keyword."]})

    books_df = artifacts.books_df
    query = query.strip()

    title_matches, author_matches, category_matches = _match_metadata(books_df, query)
    matched_books = books_df[title_matches | author_matches | category_matches]

    if title_matches.any():
        seed_index = books_df[title_matches].index[0]
        similarity_scores = list(enumerate(artifacts.similarity_matrix[seed_index]))
        sorted_scores = sorted(similarity_scores, key=lambda item: item[1], reverse=True)
        recommended_indices = [
            idx for idx, _ in sorted_scores if idx != seed_index
        ][:top_n]
    elif author_matches.any() or category_matches.any():
        metadata_results = matched_books.sort_values(
            ["rating_score", "Title"], ascending=[False, True]
        ).head(top_n)
        recommendations = metadata_results[
            ["Title", "authors", "categories", "rating_score", "top_reviews"]
        ].copy()
        recommendations = recommendations.rename(
            columns={
                "authors": "Author",
                "categories": "Genre",
                "rating_score": "Rating Score",
                "top_reviews": "Top Reviews",
            }
        )
        recommendations.insert(0, "Query", query)
        return recommendations.reset_index(drop=True)
    else:
        processed_query = preprocess_text(query)
        if not processed_query:
            return pd.DataFrame({"message": ["No recommendations found for this query."]})
        query_vector = artifacts.vectorizer.transform([processed_query])
        query_scores = cosine_similarity(query_vector, artifacts.tfidf_matrix).flatten()
        recommended_indices = np.argsort(query_scores)[::-1][:top_n]
        if len(recommended_indices) == 0 or query_scores[recommended_indices[0]] == 0:
            return pd.DataFrame({"message": ["No recommendations found for this query."]})

    recommendations = books_df.iloc[recommended_indices][
        ["Title", "authors", "categories", "rating_score", "top_reviews"]
    ].copy()
    recommendations = recommendations.rename(
        columns={
            "authors": "Author",
            "categories": "Genre",
            "rating_score": "Rating Score",
            "top_reviews": "Top Reviews",
        }
    )
    recommendations.insert(0, "Query", query)
    return recommendations.reset_index(drop=True)
