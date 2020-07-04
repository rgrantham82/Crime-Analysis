#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis
# In between this notebook, and the first, I cleaned the data further in Excel since the dataset was small enough to begin with. The resulting dataset lists 56 separate alleged hate crimes, in Austin, TX, since 2017. Out of the total number of reported, alleged incidents, 32.14% were directed at the LGBT Community. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


# Importing & examining the cleaned dataset
df = pd.read_csv(r"C:\Users\Robert\OneDrive\Desktop\aus_final_clean.csv")
display(df.head())
print('----------------------------------')
print(df.info())
print('----------------------------------')
print(df.isnull().sum())


# In[3]:


# Creating an index from the dates and performing the necessary conversions
df['date'] = df['date'].astype('datetime64')
df = df.set_index('date')
df['bias'] = df['bias'].astype('category')
df['offense'] = df['offense'].astype('category')
df['offense_location'] = df['offense_location'].astype('category')
df['offender_ethnicity'] = df['offender_ethnicity'].astype('category')

# Reexamining the dataset
print(df.info())
print('----------------------------------')
print(df.index)
print('----------------------------------')
display(df.head())
print('----------------------------------')
display(df.tail())


# We'll look at the numerical data in the victims & offenders columns later. First, I want to explore the categorical data.
# 
# ### Question 1. How are reported incidences in Austin distributed according to motivation? 

# In[4]:


bias = df.bias.value_counts()
bias_pct = df.bias.value_counts(normalize=True)
print(bias)
print('----------------------------------')
print('Total number of reported hate crimes since 2017 = 56')
print('----------------------------------')
print('Incident Biases as Percentages:')
print(bias_pct)

bias.plot.bar()
plt.xlabel('Motivation')
plt.ylabel('Total')
plt.title('Distribution of Incidents according to Motivation')
plt.show()


# ### Question 2. How are the offense-types distributed? 

# In[5]:


offense_count = df.offense.value_counts()
display(offense_count)
print('----------------------------------')
print('Offense Percentages:')
display(df.offense.value_counts(normalize=True))


offense_count.plot.bar()

plt.xlabel('Offense')
plt.ylabel('Total')
plt.title('Distribution of Offenses')
plt.show()


# ### Question 3. How are the offenders distributed according to race/ethnicity?      

# In[6]:


print(df.number_of_offenders_under_18.sum())
print(df.number_of_offenders_over_18.sum())
print('----------------------------------')
print('Total # of offenders = 64')


# In[7]:


offenders_count = df['offender_ethnicity'].value_counts()
print(offenders_count)
print('----------------------------------')
print('Percentages of offender(s) ethnicity:')
print(df.offender_ethnicity.value_counts(normalize=True))

offenders_count.plot.bar()

plt.xlabel('Ethnicity')
plt.ylabel('Total')
plt.title('Distribution of Offender(s) Ethnicity')
plt.show()


# Note...the above 'Offender' graph has an instance of 'Hispanic (2), Caucasian (2)' as a single column because of an incident that occurred on 1/19/19 https://www.statesman.com/news/20200124/confrontation-that-ignited-attack-on-austin-gay-couple-questioned-by-detective -- 2 of the offenders were white, and the other 2 were hispanic. 

# ### Question 4. How are the offenses (type) distributed? 

# In[8]:


# Examining the 'offense location' column
location = df.offense_location.value_counts()
print(location)
print('----------------------------------')
print(df.offense_location.value_counts(normalize=True))

location.plot.bar()
plt.show()


# Interestingly, the 2nd highest percentage of crimes take place within the home (19.64%).
# 
# The victim columns only contain numbers and no indicators of the victims' ethnicity/race. Really, all we can do with these columns is see how they correlate with the offender columns. 

# ### Question 5. Any correlations? 

# In[9]:


# Examining correlations between victims & offenders
df_corr = df.corr()

display(df_corr)
df_corr.plot.bar()
plt.show()

