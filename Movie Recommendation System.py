#!/usr/bin/env python
# coding: utf-8

# In this project, we are going to build a simple movie recommendation system using Python and pandas.

# The system will suggest movies based on their similarity to other movies. While it's a basic model, it's a great starting point for understanding recommender systems.

# In[1]:


import numpy as np
import pandas as pd


# We'll work with two datasets:
# 
# - User ratings for movies.
# - Movie titles and their IDs.

# In[29]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('/Users/sowmya/Downloads/u.data', sep='\t', names=column_names)


# In[30]:


df.head()


# Reading the movie tiles

# In[31]:


movie_titles = pd.read_csv("/Users/sowmya/Downloads/Movie_Id_Titles.txt")
movie_titles.head()


# we can merge them

# In[33]:


df = pd.merge(df,movie_titles,on='item_id')
df.head()


# ### Exploratory Data Analysis

# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# let's construct a DataFrame that holds the average rating alongside the total count of ratings for each item.

# In[35]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[36]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[37]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()


# Setting the number of ratings column:

# In[39]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()


# Visualizing the number of ratings:

# In[41]:


plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=40)


# In[42]:


plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=40)


# It makes intuitive sense for most ratings to be around the 3.0 mark.

# ### Recommending Similar Movies

# Next, we'll create a matrix with user IDs on one axis and movie titles on the other. Each cell in this matrix will show how a user rated a specific movie.

# In[44]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


# It's common to see many NaN values because not every user has rated most of the movies.
# 
# Let's find out which movie has the most ratings.

# In[45]:


ratings.sort_values('num of ratings',ascending=False).head(10)


# We'll concentrate on two specific films: "Star Wars," representing the sci-fi genre, and "Dumb and Dumber," embodying comedy.

# In[46]:


ratings.head()


# Now let's grab the user ratings for those two movies:

# In[47]:


starwars_user_ratings = moviemat['Star Wars (1977)']
dumb_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


# We can then use corrwith() method to get correlations between two pandas series:

# In[48]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(dumb_user_ratings)


# We'll clean up the data by getting rid of the NaN values and switching to a DataFrame from a series.

# In[49]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()


# Sorting the dataframe by correlation will show us movies similar to "Star Wars." However, some results might be off because many movies were only watched by people who saw "Star Wars," the most popular movie, just once.

# In[50]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# Let's improve our results by only including movies with at least 100 reviews, a decision based on the earlier histogram data.

# In[51]:


corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# Now sort the values and notice how the titles make a lot more sense

# In[53]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# Now the same for Liar Liar:

# In[54]:


corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# ### Our simple recommendation system worked: It suggested other "Star Wars" movies and a George Lucas film for "Star Wars" fans. For "Liar Liar," there's room for improvement, but it did recommend another Jim Carrey movie.
