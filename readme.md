### movie recommender built with scikit learn vector distance and streamlit frontend

1. import csvs & data cleaning
2. uses countvectorizer to convert each movie’s tags into a fixed-length numeric vector (excluding english stopwords)
3. calculates cosine similarity between all movie vectors which measures how textually similar any two movies are
4. uses tmdb api for poster generation