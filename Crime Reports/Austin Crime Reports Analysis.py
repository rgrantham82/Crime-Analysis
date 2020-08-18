#!/usr/bin/env python
# coding: utf-8

# # Analyzing Austin PD's Crime Reports Dataset
# 
# The dataset is available from the Austin Police Department on https://data.austintexas.gov/Public-Safety/Crime-Reports/fdj4-gpfu. It is updated weekly and I last downloaded the dataset on 8/10/2020.  
# 
# 
# 
# ## Table of Contents 
# 
#     I. Introduction
#     II. Data Scrubbing
#     III. Exploratory Analysis and Visualizations 
#     IV. Data Modeling
#     
#     Questions:
# ><ul>
# ><li><a href="#q1"> 1. What areas of Austin have the highest crime rates?</a></li>
# ><li><a href="#q2"> 2. How is crime distributed in 78753?</a></li> 
# ><li><a href="#q3"> 3. How is crime distributed in 78741?</a></li>
# ><li><a href="#q4"> 4. How are violent crimes, in particular murder and rape, distributed?</a></li>

# ## I. Introduction
# 
# I began reviewing the Crime Reports dataset, provided by the Austin PD, around the same time I began reviewing its Hate Crimes datasets for analysis, at the beginning of 2020. This is a rather large dataset, containing over 2 million records, spanning from 2003 to the present, and is update weekly. 
# 
# This is a self-paced project, conceived outside of both work and the educational arenas. It is my hope that this project will reveal some actionable insights that will benefit the Austin law enforcement community, news outlets, and anyone else interested in gaining knowledge on how best to combat the problem of crime in the Austin area.
# 
# I originally attempted importing the data into this notebook using Sodapy's Socrata API method but found it cumbersome. Mainly, it didn't want to work with importing the entire dataset, and added several redundant columns. I, therefore, prefer to manually download the entire dataset and re-download each week after it's updated.

# In[1]:


# Importing essential libraries and configurations
import contextily as cxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

get_ipython().magic('matplotlib inline')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# In[2]:


# Loading the data
df = pd.read_csv('crime_reports.csv')

# Making a copy to work with later
df2 = df.copy()


# In[3]:


# Examining the dataframe
display(df.head())
print('----------------------------------')
display(df.tail())
print('----------------------------------')
display(df.info())
print('----------------------------------')
display(df.duplicated().sum())
print('----------------------------------')
display(df.isnull().sum())


# ## II. Data Scrubbing
# 
# There are several columns of data we won't be using in the analysis, mainly other date and geodata columns. So we'll drop those and also scrub some others. Mainly, we want the zip code and address columns to be free of nulls and duplicates. 

# In[4]:


# Create a helper function to aid in data scrubbing
def clean_data(df):
    """Deleting unnecessary columns"""
    drop_col = ['Occurred Date', 'Occurred Time', 'Report Date', 'Report Time', 'Census Tract', 'UCR Category', 
                'Category Description', 'X-coordinate', 'Y-coordinate', 'Location'] 
    df.drop(drop_col, axis=1, inplace=True) 
    """Cleaning essential columns and/or converting their datatype"""
    clean_col = ['Occurred Date Time', 'Zip Code'] 
    df.dropna(subset=clean_col, inplace=True)
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    df.zip_code = df.zip_code.astype('int64')
    df.clearance_date = df.clearance_date.astype('datetime64') 
    df.highest_offense_description = df.highest_offense_description.astype('category') 
    df.location_type = df.location_type.astype('category') 
    df.apd_sector = df.apd_sector.astype('category') 
    df.report_date_time = df.report_date_time.astype('datetime64') 
    df['month'] = df['report_date_time'].dt.month
    """Setting the index"""
    df.set_index(['report_date_time'], inplace=True) 
    df.sort_index(inplace=True)
    return df
df = clean_data(df)


# I recently discovered that after the initial reshaping and cleaning of a dataframe, when beforehand there were none, duplicates will somehow show. So, to be on the safe side, we are going to recheck for duplicates before moving ahead with exploratory analysis. 

# In[5]:


# Rechecking the dataframe 
display(df.isnull().sum())
print('----------------------------------')
display(df.dtypes)
print('----------------------------------')
display(df.head())
print('----------------------------------')
display(df.tail())


# ## III. Exploratory Analysis

# <a id='q1'></a>
# ### A. Question 1. What areas of Austin have the highest crime rates? 
# 
# ***Note: I am only including zipcodes and crimes, for questions 1 - 3, that >= 1%. Any zipcodes or crime percentages, below 1%, will be discluded to simplify analysis and visualizations.***

# In[10]:


# Create and show dataframe for crime rates by zipcode
zip_codes = df.zip_code.value_counts().head(24)
display(zip_codes)

# Showing the results as percentages
print('----------------------------------')
display(df.zip_code.value_counts(normalize=True).head(24))

zip_codes.plot.bar(figsize=(20,10), fontsize=14, rot=60)

plt.xlabel('Zip Code')
plt.ylabel('Police Reports Taken')
plt.title('Crime Rate by Zipcode')


# Out of all the areas in Austin, 78741 has the highest percentage of overall crime at 9.08%. This is a significant 1.3 percentage points higher than the number 2 area 78753 which hosts 7.78% of overall crime.

# #### Taking a closer look at particular areas... 
# 
# Because 78753 is my resident zipcode, I chose to examine it first. 
# 
# Next, I'll examine 78741. 

# <a id='q2'></a>
# ### B. Question 2. How is crime distributed in 78753? 

# In[11]:


# Examining crime in the 78753 area
df_53 = df[df.zip_code == 78753]

# Create a dataframe for the top 10 crime categories in the zipcode
df_53_off = df_53.highest_offense_description.value_counts().head(22)

# Display the different crime values & then as percentages 
display(df_53_off)
print('----------------------------------')
display(df_53.highest_offense_description.value_counts(normalize=True).head(22))

df_53_off.plot.pie(figsize=(10,10), fontsize=14, rot=60)
plt.title('Crime Distribution (78753)')


# <a id='q3'></a>
# ### C. Question 3. How is crime distributed in 78741? 

# In[12]:


# Create a dataframe for crime in the 78741 area (the highest amount of crime of any Austin zip code)
df_41 = df[df.zip_code == 78741]

# Create a dataframe for the top 10 crime categories in the zipcode
df_41_off = df_41.highest_offense_description.value_counts().head(21)

# Display the different crime values & then as percentages 
display(df_41_off)
print('----------------------------------')
display(df_41.highest_offense_description.value_counts(normalize=True).head(21))

df_41_off.plot.pie(figsize=(10,10), fontsize=14)
plt.title('Crime Distribution (78741)')


# <a id='q4'></a>
# ### D. Question 4. How are violent crimes, in particular murder, capital murder, and rape, distributed? 
# 
# ***Note: Only including areas where rape crimes >= 1%.***

# In[13]:


# Create a dataframe for murders, capital murders, and rapes
df_mur = df[df.highest_offense_description == 'MURDER']
df_mur_cap = df[df.highest_offense_description == 'CAPITAL MURDER']
df_rape = df[df.highest_offense_description == 'RAPE']
df_agg_asslt = df[df.highest_offense_description == 'AGG ASSAULT']

# What are the top zipcodes for murders? 
print('----------------------------------')
print('Murder')
print('----------------------------------')
df_mur_val = df_mur.zip_code.value_counts().head(24)
display(df_mur_val)
print('----------------------------------')
display(df_mur.zip_code.value_counts(normalize=True).head(24))

# What are the top zipcodes for capital murders? 
print('----------------------------------')
print('Capital Murder')
print('----------------------------------')
df_mur_cap_val = df_mur_cap.zip_code.value_counts()
display(df_mur_cap_val)
print('----------------------------------')
display(df_mur_cap.zip_code.value_counts(normalize=True))

# What are the top 10 zipcodes for rape? 
print('----------------------------------')
print('Rape')
print('----------------------------------')
df_rape_val = df_rape.zip_code.value_counts().head(21)
display(df_rape_val)
print('----------------------------------')
display(df_rape.zip_code.value_counts(normalize=True).head(21))

print('----------------------------------')
print('Aggrivated Assault')
print('----------------------------------')
df_agg_asslt_val = df_agg_asslt.zip_code.value_counts().head(16)
display(df_agg_asslt_val)
print('----------------------------------')
display(df_agg_asslt.zip_code.value_counts(normalize=True).head(16))
        

df_mur_val.plot.bar(figsize=(20,10), rot=60, fontsize=14)
plt.title('Murder')
plt.show()

df_mur_cap_val.plot.bar(figsize=(20,10), rot=60, fontsize=14)
plt.title('Capital Murder')
plt.show()

df_rape_val.plot.bar(figsize=(20,10), fontsize=14, rot=60)
plt.title('Rape')
plt.show()

df_agg_asslt_val.plot.bar(figsize=(20,10), fontsize=14, rot=60)
plt.title('Aggrivated Assault')
plt.show()


# #### Showing Murder, Capital Murder, Rape, and Aggrivated Assault on the map...
# 
# ***Note: I attempted to use the Contextily library in order to add a basemap to the following plots but was unsuccessful.***

# In[14]:


df_viol = df.query('highest_offense_description == ["MURDER", "CAPITAL MURDER", "RAPE", "AGG ASSAULT"]') 

df_viol_zip = df_viol.zip_code.value_counts().head(17)
display(df_viol_zip)
display(df_viol.zip_code.value_counts(normalize=True).head(17))

df_viol_zip.plot.bar(figsize=(20,10), fontsize=14, rot=60)
plt.title('Violent Crime Distribution since 2003')
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(df_viol['longitude'], df_viol['latitude'], 'o', markersize=1)
plt.show()


# In[45]:


viol_month = df_viol.groupby(df_viol['month'])['highest_offense_description'].value_counts()
display(viol_month)
display(df_viol.groupby(df_viol['month'])['highest_offense_description'].value_counts(normalize=True))

fig, axs = plt.subplots(figsize=(20,20))
df_viol.groupby(df_viol['month'])['highest_offense_description'].value_counts().plot.barh(ax=axs)
plt.show()


# In[ ]:


viol_month.to_csv('datasets\viol_month.csv')


# In[15]:


viol_freq = pd.crosstab(df_viol.zip_code, df_viol.highest_offense_description)
display(viol_freq)


# In[59]:


viol_freq.to_csv('datasets\viol_freq.csv')


# In[46]:


viol_freq_1 = viol_freq.loc['78701':'78720']
viol_freq_2 = viol_freq.loc['78721':'78730']
viol_freq_3 = viol_freq.loc['78731':'78740']
viol_freq_4 = viol_freq.loc['78741':'78749']
viol_freq_5 = viol_freq.loc['78750':'78760']

viol_freq_1.plot.bar(figsize=(20,10), stacked=True, fontsize=14, rot=60)
plt.show()

viol_freq_2.plot.bar(figsize=(20,10), stacked=True, fontsize=14, rot=60)
plt.show()

viol_freq_3.plot.bar(figsize=(20,10), stacked=True, fontsize=14, rot=60)
plt.show()

viol_freq_4.plot.bar(figsize=(20,10), stacked=True, fontsize=14, rot=60)
plt.show()

viol_freq_5.plot.bar(figsize=(20,10), stacked=True, fontsize=14, rot=60)
plt.show()


# In[17]:


viol_freq.corr()


# So far, 78753 and 78741 are the top hotspots for all sorts of crime in Austin, including violent crime.
# 
# For non-capital murder, 78741 comes in at number 1 with 10.91%. 
# 
# #### ***It is important to note that murder does not necessarily make the defendant(s) automatically eligible for the death penalty. Under Texas law, we distinguish capital murder, through the motives and actions of the defendant(s) during the commission of a homicide, as whether or not automatically warranting an eventual date with the executioner. This includes such things as if the homicide was premeditated or not, if the defendant(s) murdered a police officer, etc.***
# 
# Regarding capital murder, 78723 comes in to share the number one spot with 78753, with a rate of 13.92% each. The 78741 area drops to the number 3 spot, carrying 8.97%.
# 
# So, if we're honest, 78753 actually is the number 1 hotspot for murder because it has played host to 40 non-capital murders and 11 capital murders in total, meaning it has hosted 11% of all murders. 78741, if we combine all murders, only accounts for 10.56% of the total since 2003. 
# 
# Next, 78741 climbs back to claim the number 1 spot for rape at 12.09% -- 3.43 percentage points higher than the number 2 spot 78753 carrying 8.66% which is quite a significant lead when you look at it on the graph!! Why does rape occur so much more often in this area than in others? 
# 
# 
# 
# In the next part of the analysis, we'll look even closer at all the above data... 

# In[58]:


df.to_csv('datasets\crime_reports_copy.csv')
df_53.to_csv('datasets\df_53.csv')
df_41.to_csv('datasets\df_41.csv')
df_viol.to_csv('datasets\df_viol.csv')

