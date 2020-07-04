#!/usr/bin/env python
# coding: utf-8

# # Analyzing hate crimes trends for Austin against the USA as a whole, 2017 - Present

# ## Data Wrangling & Cleaning

# I've been working, off and on, on this project for since about January 2020. One-half practice, one-half because I want to try and contribute to making sense of the chaos that is our world right now. What I intend is to analyze hate crimes trends for Austin, TX against the USA as a whole from 2017 to the present, with particular focus on the LGBT Community. 
# 
# I am using data provided by Austin PD in this notebook, and in the next 2, or 3 notebooks as well. For now, I am focusing solely on data for Austin. I will get into broader data for the USA later down the road.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

aus_17 = pd.read_csv('https://data.austintexas.gov/resource/79qh-wdpx.csv')
display(aus_17.head())
print('----------------------------------')
display(aus_17.dtypes)


# I dislike the Socrata method bc it imports all data in all columns as objects...importing the data using the url method leaves the column data intact so will make my job much easier down the road. 

# ### First glance...
# As I stated previously, my goal is to analyze trends over time. In particular, I want to focus on how hate crime affects the LGBT community. Initially speaking, most of these columns will be unnecessary for my purposes so I suspect we'll be removing most of them. 

# In[2]:


# Loading the datasets for '18, '19, and this year
aus_18 = pd.read_csv('https://data.austintexas.gov/resource/idj2-d9th.csv')
aus_19 = pd.read_csv('https://data.austintexas.gov/resource/e3qf-htd9.csv')
aus_20 = pd.read_csv('https://data.austintexas.gov/resource/vc9m-ha4y.csv')


# In[3]:


# Concatenating the datasets
aus_final = pd.concat([aus_17, aus_18, aus_19, aus_20], sort=False, axis=0)

# Exploring the concatenated set
display(aus_final.head())
print('----------------------------------')
display(aus_final.tail())
print('----------------------------------')
display(aus_final.isnull().sum())


# We can still go a bit farther in the cleaning process:
# 
# 1. The 'incident_number' column can be split along the '-' -- we can name a new 'year' column and convert it into datetime, and we can create a new 'incident_number' column. 
# 2. There are various descriptions in the 'bias' column that can be categorized into one variable as 'anti-lgbt.' Let's see what we can do with these. 
# 3. Also, we can convert the 'bias' column into a category type.

# What I want to do now is split the 'incident_number' column along the '-' because the #s before the '-' clearly indicate the year the incident takes place, which I want to merge with the corresponding months in the 'month' column, and keep the numbers after the '-' as the 'incident_number.' 

# In[4]:


# It took me a few tries but I think I finally got it figured out! Now for the final push! Let's pray everybody! :P 
new = aus_final["incident_number"].str.split("-", n = 1, expand = True)
aus_final["year"]= new[0]
aus_final["occurence_number"]= new[1]
aus_final.drop(columns =["incident_number"], inplace = True)
aus_final['date'] = aus_final[['month', 'year']].agg('-'.join, axis=1)
aus_final.drop(['month', 'occurence_number', 'year'], axis=1, inplace=True)
aus_final = aus_final[['date', 'bias', 'number_of_victims_over_18', 'offense_location']]
aus_final.rename(columns={'number_of_victims_over_18': 'victims'}, inplace=True)
aus_final['date'] = pd.to_datetime(aus_final['date'])
aus_final.set_index('date', inplace=True)

# Showing the final product
display(aus_final.head())
print('----------------------------------')
display(aus_final.tail())
print('----------------------------------')
display(aus_final.isnull().sum())
print('----------------------------------')
display(aus_final.describe())
print('----------------------------------')
display(aus_final.columns)
print('----------------------------------')
display(aus_final.index)


# In[5]:


aus_final.plot()
plt.show()


# In[8]:


bias = aus_final.bias.value_counts(normalize=True)
print(bias)
bias.plot.bar()
plt.show()

