# Book Recommendation System

This project is an Information Retrieval and Text Mining (IRTM) based **book recommendation system** built with **Python** and **Streamlit**. It recommends books by analyzing book metadata and user review text using **TF-IDF** and **cosine similarity**.

The system allows users to search by **book title**, **author name**, **genre/category**, or **free-text keywords**. It then returns the most relevant books along with useful details such as author, genre, rating, and top reviews.

## Project Highlights

- Content-based recommendation system
- Search support for title, author, genre, and keywords
- Uses TF-IDF vectorization for text representation
- Uses cosine similarity to find related books
- Displays top 5 recommendations
- Shows average rating and top 3 reviews for each book
- Streamlit web interface for interactive use
- Fetches book cover images using the Google Books API

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Requests

## How It Works

The recommendation pipeline follows these steps:

1. Loads the book review dataset from a CSV file
2. Cleans and preprocesses the text data
3. Combines title, author, description, categories, review summary, and review text into one content field
4. Applies text preprocessing such as:
   - lowercasing
   - punctuation removal
   - stopword removal
   - lemmatization
5. Converts text into numerical vectors using **TF-IDF**
6. Computes similarity scores using **cosine similarity**
7. Returns the most relevant books based on the user query

## Recommendation Modes

The app supports multiple query types:

- **Title search**: finds similar books based on the selected book
- **Author search**: returns highly rated books by the matching author
- **Genre/category search**: returns highly rated books from the matching category
- **Keyword search**: uses processed text similarity to recommend relevant books

## Project Structure

```text
Irtm_Project/
├── README.md
└── Project_Code/
    ├── app.py
    ├── book_recommender.py
    ├── requirements.txt
    └── Book_Recommendation_System.ipynb
```

## Dataset

The dataset file is not included in this repository because it is large.

Expected dataset file:

```text
Project_Code/anotherOne.csv
```

The application expects this CSV to contain fields used by the project, such as:

- `Title`
- `authors`
- `publisher`
- `description`
- `categories`
- `review/summary`
- `review/text`
- `review/score`

If you are using the Amazon Books reviews dataset or a processed version of it, place the final CSV file at the path shown above.

## Installation

1. Clone this repository:

```bash
git clone <your-repository-url>
cd Irtm_Project
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r Project_Code/requirements.txt
```

## Running the Application

Start the Streamlit app with:

```bash
streamlit run Project_Code/app.py
```

After starting, open the local Streamlit URL shown in your terminal, usually:

```text
http://localhost:8501
```

## Application Features

- Simple and interactive Streamlit interface
- Query input box for user search
- Top book recommendations displayed in card format
- Book cover images fetched dynamically
- Detailed view for each recommended book
- Top reviews shown for better understanding of recommendations

## NLTK Resources

The project automatically downloads the required NLTK resources if they are not already available. These include:

- `stopwords`
- `punkt`
- `wordnet`
- `omw-1.4`

## Output

For each recommended book, the system can display:

- Book title
- Author
- Genre/category
- Average rating score
- Top 3 reviews
- Cover image when available

## Use Cases

- Personalized book discovery
- Book similarity search
- Genre-based exploration
- Author-based recommendation
- Keyword-based content retrieval

## Future Improvements

- Add collaborative filtering
- Improve ranking with hybrid recommendation techniques
- Add filters for publisher, rating, and year
- Deploy the application online
- Save user search history and favorites

## Author

Developed as part of an **IRTM (Information Retrieval and Text Mining)** project.
