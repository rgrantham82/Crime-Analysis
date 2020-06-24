#!/usr/bin/env python
# coding: utf-8

# # Further Cleaning & Exploratory Analysis
# In between this notebook, and the first, I cleaned the data further in Excel since the dataset was small enough to begin with. The resulting dataset lists 56 separate alleged hate crimes, in Austin, TX, since 2017. Out of the total number of reported incidents, 32.14% were allegedly directed at the LGBT Community. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


# Importing the cleaned dataset
df = pd.read_csv(r"C:\Users\Robert\OneDrive\Desktop\aus_final.csv")
display(df.info())


# ## Convert 'date_of_inciedent' to datetime64 and make it the index

# In[3]:


df.date_of_incident = df.date_of_incident.astype('datetime64')
df = df.set_index('date_of_incident')


# In[4]:


display(df.head())
print('----------------------------------')
display(df.shape)
print('----------------------------------')
display(df.index)


# ## Not much we can do with the 'victims' and 'offenders' columns just yet...
# We have much more categorical data we can work with first. 
# 
# ### Question 1. How are reported incidences in Austin distributed according to motivation? 

# In[5]:


bias = df.motivation.value_counts()
print('total number of reported hate crimes since 2017:')
display(bias.sum())
print('----------------------------------')
display(bias)

bias.plot.bar()
plt.xlabel('Motivation')
plt.ylabel('Total')
plt.title('Distribution of Incidents according to Motivation')
plt.show()


# #### I am unsure why the Anti-African American category is splitting into two. Regardless, it still maintains its accuracy.

# ## Question 2. How are the offense-types distributed? 

# In[6]:


offense_count = df.offense.value_counts()
display(offense_count)
offense_count.plot.bar()
plt.xlabel('Offense')
plt.ylabel('Total')
plt.title('Distribution of Offense Types')
plt.show()


# ## Question 3. How are the alleged offenders distributed according to race? 

# In[7]:


offenders_count = df['race_of_offender(s)'].value_counts()
display(offenders_count)
offenders_count.plot.bar()
plt.xlabel('Ethnicity')
plt.ylabel('Total')
plt.title('Distribution of Offender(s) Ethnicity')
plt.show()

