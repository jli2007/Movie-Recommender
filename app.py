import streamlit as st
import pickle
import pandas as pd
import requests

apiKey = st.secrets["tmdb"]["api"]
movies_list = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_list)
similarity = pickle.load(open('similarity.pkl', 'rb'))

def fetchPoster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={apiKey}')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500" + data['poster_path']
    
    
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_movies_posters=[]
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies_posters.append(fetchPoster(movie_id))
        
    return recommended_movies, recommended_movies_posters
    
    
st.title("movie recommender system")

selected_movie = st.selectbox('select movie', movies['title'].values)    

if st.button('find recommendations'):
    movies, posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5) 
    
    columns = [col1, col2, col3, col4, col5]

    for i in range(5):
        with columns[i]:
            st.text(movies[i])
            st.image(posters[i])