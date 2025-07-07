#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) 
pd.set_option("display.max_colwidth", None)


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.head(1)


# In[5]:


credits.head(1)['cast'].values


# In[6]:


movies = movies.merge(credits,on='title')


# In[7]:


movies.head().iloc[0]
# things from credit are merged in


# In[8]:


movies['original_language'].value_counts()
# dict 


# In[9]:


movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# In[10]:


movies.head().iloc[0]


# In[11]:


movies.isnull().sum()


# In[12]:


movies.dropna(inplace=True)
# line above initally had null values, ran it again


# In[13]:


movies.duplicated().sum()
# no duplicates yessir


# In[14]:


movies.iloc[0].genres


# In[15]:


# ['Action', 'Adventure', 'Fantasy', 'SciFi']
import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# get "top 3 actors"
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[16]:


movies['genres'] = movies['genres'].apply(convert)
# every movie, get all genres for the movies
# movies['genres'] = so it doesnt generate output


# In[17]:


movies.head().iloc[0]


# In[18]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[19]:


movies.head().iloc[0]


# In[20]:


movies['cast'][0]


# In[21]:


movies['cast'] = movies['cast'].apply(convert3)


# In[22]:


movies.head().iloc[0]


# In[23]:


# we want the director from crew
import ast
def get_crew(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[24]:


movies['crew'] = movies['crew'].apply(get_crew)


# In[25]:


movies.head()


# In[26]:


movies['overview'][0]


# In[27]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[28]:


# we want to remove spaces from words as if we want "sam" alot of ppl are returned
# sam worthington -> samworthington

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[29]:


# this is optimal format to combine - all in arrays 
movies.head().iloc[0]


# In[30]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[31]:


movies.head().iloc[0]


# In[32]:


new_df = movies[['movie_id', 'title', 'tags']]


# In[33]:


new_df.iloc[0]


# In[34]:


# commas into spaces basically
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[35]:


new_df['tags'][0]


# In[36]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[37]:


new_df.head()


# In[38]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[39]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


# In[40]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[41]:


# text -> vectors then find closest vectors approach
# scikit learn as vector functions
# we first want to remove stop words like "a", "or", etc because they are useless
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[42]:


vectors = cv.fit_transform(new_df['tags']).toarray() # type: ignore


# In[43]:


# first movie
vectors[0]


# In[44]:


cv.get_feature_names_out()


# In[45]:


ps.stem('loved')


# In[46]:


from sklearn.metrics.pairwise import cosine_similarity


# In[47]:


similarity = cosine_similarity(vectors)


# In[48]:


sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])


# In[49]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[50]:


recommend('Batman Begins')


# In[ ]:


import pickle
def generatePKL():
    pickle.dump(new_df, open('movie_dict.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl','wb'))


# In[53]:


new_df.head().iloc[0].movie_id


# In[ ]:


if __name__ == "__main__":
    generatePKL()

