#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv("D:\\tmdb_5000_movies.csv")
credits = pd.read_csv("D:\\tmdb_5000_credits.csv")


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movies = movies.merge(credits,on='title')  


# In[6]:


movies.head(1)


# In[7]:


movies.info()


# In[8]:



# moies_id
# title
# gerner
id 
#keywords
#title 
#Overviwe
#cast
#crow
movies= movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.head(5)


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ['Action','adventure','Fantasy','Scifi']
import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')



# In[15]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L  


# In[16]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[17]:


movies['keywords']= movies['keywords'].apply(convert)


# In[18]:


movies.head()


# In[ ]:





# In[19]:


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


# In[20]:


movies['cast'].apply(convert3)


# In[21]:


movies['cast'] = movies['cast'].apply(convert3)


# In[22]:


movies.head()


# In[23]:


movies['crew'][0]


# In[24]:


def fetch_director(obj):
    L =[]
    for i in ast.literal_eval(obj):
        if i ['job']=='Director':
            L.append(i['name'])
            break
    return L

            


# In[25]:


movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()


# In[26]:


movies['overview'][0]


# In[27]:


movies['overview']= movies['overview'].apply(lambda x:x.split())


# In[28]:


movies.head()


# In[29]:


# 'Sam Worthington'
# 'SamWorthington'
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for  i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for  i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for  i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for  i in x])


# In[30]:


movies.head()


# In[31]:


movies['tags']=movies['overview'] + movies['genres'] +movies['keywords'] +movies['cast'] +movies['crew']
    
    


# In[32]:


movies.head()


# In[33]:


new_df = movies[['movie_id','title','tags']]
new_df


# In[34]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[35]:


new_df['tags'][0]


# In[36]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[37]:


new_df.head()


# In[38]:


import nltk


# In[39]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[40]:


def stem(text):
    y= []
     
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[41]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[42]:


new_df['tags'][0]


# In[43]:


new_df['tags'][1]


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english') 


# In[45]:


vector = cv.fit_transform(new_df['tags']).toarray() #word transformation 


# In[46]:


vector


# In[47]:


len(cv.get_feature_names_out())


# In[ ]:





# In[48]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[49]:


from sklearn import metrics


# In[50]:


from sklearn.metrics.pairwise import cosine_similarity


# In[51]:


similarity = (cosine_similarity(vector))


# In[52]:


sorted(list(enumerate(similarity[1])),reverse=True,key=lambda x:x[1])[1:6]


# In[53]:


def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(i[0])


# In[54]:


recommend('Batman Begins')


# In[55]:


new_df.iloc[65].title


# In[56]:


import pickle


# In[57]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[58]:


new_df['title'].values


# In[59]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[60]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




