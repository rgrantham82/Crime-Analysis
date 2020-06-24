#!/usr/bin/env python
# coding: utf-8

# # Analyzing hate crimes trends for Austin against the USA as a whole, 2017 - Present

# ## Data Wrangling & Cleaning

# I've been working, off and on, on this project for since about January 2020. One-half practice, one-half because I want to try and contribute to making sense of the chaos that is our world right now. What I intend is to analyze hate crimes trends for Austin, TX, from 2017 to the present, with particular focus on the LGBT Community. 
# 
# I am using data provided by Austin PD in this notebook, and in the next 2, or 3 notebooks as well. For now, I am focusing solely on data for Austin. I will get into broader data for the USA later down the road.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
display(aus_final.head())
print('----------------------------------')
display(aus_final.tail())
print('----------------------------------')
display(aus_final.describe())
print('----------------------------------')
display(aus_final.info())
print("-------------------------------")
display(aus_final.isnull().sum())


# In[4]:


aus_final.to_csv(r"C:\Users\Robert\OneDrive\Desktop\datasets\aus_final.csv")

