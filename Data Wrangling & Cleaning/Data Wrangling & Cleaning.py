#!/usr/bin/env python
# coding: utf-8

# # Analyzing hate crimes trends for Austin, 2017 - Present

# ## Data Wrangling & Cleaning

# I've been working on this project for since about January 2020. One-half keeping my coding skills sharp, one-half because I want tocontribute to making sense of the chaos that is our world right now. What I intend is to analyze hate crimes trends for Austin, TX against the USA as a whole from 2017 to the present, with particular focus on the LGBT Community. 
# 
# I am using data provided by Austin PD in this notebook, and in the next 2, or 3 notebooks as well. For now, I am focusing solely on data for Austin. I will get into broader data for the USA later down the road.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


# Loading and examining the 2017 hate crimes dataset
aus_17 = pd.read_csv('https://data.austintexas.gov/resource/79qh-wdpx.csv')
display(aus_17.head())
print('----------------------------------')
display(aus_17.dtypes)


# I dislike the Socrata method of importing data because every column in a dataset imports as an object...importing the data using the url method leaves the column data intact so will make my job much easier down the road. 
# 
# ### First glance...
# As I stated previously, my goal is to analyze trends over time. In particular, I want to focus on how hate crime affects the LGBT community. Initially speaking, most of these columns will be unnecessary for my purposes so I suspect we'll be removing most of them. 

# In[3]:


# Loading the datasets for '18, '19, and 2020
aus_18 = pd.read_csv('https://data.austintexas.gov/resource/idj2-d9th.csv')
aus_19 = pd.read_csv('https://data.austintexas.gov/resource/e3qf-htd9.csv')
aus_20 = pd.read_csv('https://data.austintexas.gov/resource/vc9m-ha4y.csv')


# In[4]:


# Concatenating the datasets
aus_final = pd.concat([aus_17, aus_18, aus_19, aus_20], sort=False, axis=0)

# Examining the new dataset
display(aus_final.head())
print('----------------------------------')
display(aus_final.tail())
print('----------------------------------')
display(aus_final.isnull().sum())


# 1. The 'incident_number' column can be split along the '-' -- we can name a new 'year' column and convert it into datetime, and we can create a new 'incident_number' column. 
# 2. There are various descriptions in the 'bias' column that can be categorized into one variable as 'anti-lgbt.' Let's see what we can do with these. 
# 3. Also, we can convert the 'bias' column into a category type.

# What I want to do now is split the 'incident_number' column along the '-' because the #s before the '-' clearly indicate the year the incident takes place, which I want to merge with the corresponding months in the 'month' column, and keep the numbers after the '-' as the 'incident_number.' 

# In[11]:


# Examining the 'bias' column
bias = aus_final.bias.value_counts()
display(bias)
print('----------------------------------')
# Displaying the bias values as proportions
display(aus_final.bias.value_counts(normalize=True))
bias.plot.bar()
plt.show()
