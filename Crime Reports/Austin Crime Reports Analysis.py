#!/usr/bin/env python
# coding: utf-8

# # Analyzing Austin PD's Crime Reports Dataset
# 
# The dataset is available from the Austin Police Department on https://data.austintexas.gov/Public-Safety/Crime-Reports/fdj4-gpfu.
# 
# 
# ## Table of Contents 
# 
#     I. Introduction
#     II. Data Scrubbing
#     III. Exploratory Analysis 
#     IV. Summary
#     
#     Questions:
# ><ul>
# ><li><a href="#q1"> 1. What areas of Austin have the highest crime rates?</a></li>
# ><li><a href="#q2"> 2. How is crime distributed in 78753?</a></li> 
# ><li><a href="#q3"> 3. How is crime distributed in 78741?</a></li>
# ><li><a href="#q4"> 4. How are violent crimes, in particular murder, capital murder, aggrivated assault, and rape distributed?
# ><li><a href="#q5"> 5. What significant does the family violence factor play, in violent crime, over time? 
# ><li><a href="#q6"> 6. How does murder appear on the map?
# </a></li>

# ## I. Introduction
# 
# I began reviewing the Crime Reports dataset, provided by the Austin PD, around the same time I began reviewing its Hate Crimes datasets for analysis, at the beginning of 2020. This is a rather large dataset, containing over 2 million records, spanning from 2003 to the present, and is updated weekly. 
# 
# This is a self-paced project, conceived outside of both work and the educational arenas. It is my hope that this project will reveal some actionable insights that will benefit the Austin law enforcement community, news outlets, and anyone else interested in gaining knowledge on how best to combat the problem of crime in the Austin area.
# 
# I originally attempted importing the data into this notebook using Sodapy's Socrata API method but found it cumbersome. Mainly, it didn't want to work with importing the entire dataset, and added several redundant columns. I, therefore, prefer to manually download the entire dataset and re-download each week after it's updated.

# In[1]:


# Importing essential libraries and configurations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins
import seaborn as sns 
import warnings

plt.style.use('classic')
get_ipython().magic('matplotlib inline')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 
              None)


# In[2]:


# Loading the data
df = pd.read_csv('crime_reports.csv')


# In[3]:


# Examining the dataframe
display(df.info())
print('----------------------------------')
display(df.duplicated().sum())
print('----------------------------------')
print(df.isnull().sum())


# ## II. Data Scrubbing
# 
# There are several columns of data we won't be using in the analysis, mainly other date and geodata columns. So we'll drop those and also scrub some others. Mainly, we want the zip code and address columns to be free of nulls and duplicates. 
# 
# The Clearance Status column contains 3 types of statuses: Y for Yes, N for No, and O which stands for "cleared by other means than arrest." Therefore, I changed the column, as well as the family_violence column, to boolean type. In other words, Y and O as True, and N as False. However, you may note that areas, where there is no clearance status at all, may or may not contain a corresponding date in the clearance date column. I am unsure how best to handle this so I am open to suggestions or advice.   

# In[4]:


# Helper function for scrubbing the data
def clean_data(df):
    drop_col = ['Occurred Time', 
                'Occurred Date', 
                'Report Date', 
                'Report Time', 
                'Census Tract', 
                'UCR Category', 
                'Category Description', 
                'X-coordinate', 
                'Y-coordinate', 
                'Location']
    df.drop(drop_col, 
            axis=1, 
            inplace=True)
    clean_col = ['Zip Code', 
                 'Report Date Time', 
                 'Occurred Date Time', 
                 'Council District', 
                 'PRA'] 
    df.dropna(subset=clean_col, 
              inplace=True)
    df.rename(columns=lambda x: x.strip().lower().replace(" ", 
                                                          "_"), 
              inplace=True)
    """Convert the following to bools"""
    d = {'Y': True, 
         'N': False}
    e = {'C': True, 
         'O': True, 
         'N': False}
    df.clearance_status = df.clearance_status.map(e)
    df.clearance_status = df.clearance_status.astype('bool')
    df.family_violence  = df.family_violence.map(d)
    df.family_violence  = df.family_violence.astype('bool') 
    """Convert the following to datetime type"""
    date_col = ['occurred_date_time', 
                'clearance_date', 
                'report_date_time'] 
    """Convert the following to category type"""
    cat_col = ['highest_offense_description', 
               'location_type', 
               'apd_sector'] 
    df[date_col] = df[date_col].astype('datetime64') 
    df[cat_col]  = df[cat_col].astype('category') 
    """Convert the following to integer type"""
    int_col     = ['zip_code', 
                   'pra', 
                   'council_district']
    df['year']  = pd.to_datetime(df['occurred_date_time'], 
                                format='%m/%d/%Y').dt.year 
    df['month'] = pd.to_datetime(df['occurred_date_time'], 
                                 format='%m/%d/%Y').dt.month 
    df['hour']  = pd.to_datetime(df['occurred_date_time'], 
                                format='%m/%d/%Y').dt.hour
    df[int_col] = df[int_col].astype('int64')
    """Set the index"""
    df.set_index(['occurred_date_time'], 
                 inplace=True)
    df.sort_index(inplace=True)
    return df
df = clean_data(df)


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

# First, let's get an overall look at crime rates and how they trend over time...

# In[6]:


# Creating and visualizing a data frame for the overall yearly crime rate since 2003
crimes_per_year = df['year'].value_counts().sort_index() 

g = sns.barplot(x=crimes_per_year.index, 
                y=crimes_per_year.values)
g.set_xticklabels(g.get_xticklabels(), 
                  rotation=60)
g.set(xlabel='Year', 
      ylabel='Crimes Reported', 
      title ='Annual Crime Rates')
plt.show()

# Overall hourly crime rates as well
crimes_per_hour = df['hour'].value_counts().sort_index()

e = sns.barplot(x=crimes_per_hour.index, 
                y=crimes_per_hour.values)
e.set_xticklabels(e.get_xticklabels(), 
                  rotation=60)
e.set(xlabel='Hour', 
      ylabel='Crimes Reported', 
      title ='Hourly Crime Rates')
plt.show()


# Between 2003 and now, crime peaked in 2008 and continued a downward trend until 2019 when it rose again. Since we're still in 2020, we have to wait until the end of the year to see what 2020 yields. 

# <a id='q1'></a>
# ### A. Question 1. What areas of Austin have the highest crime rates? 
# 
# ***Note: I am only including zipcodes and crimes, for questions 1 - 3, that >= 1%. Any zipcodes or crime percentages, below 1%, will be discluded to simplify analysis and visualizations.***
# 
# Question 4 regards violent crime. For violent crime, I chose to examine 4 categories: aggrivated assault, rape, murder, and capital murder. I realize there are other types of violent crime, but for now I am sticking with these 4 categories. 

# In[7]:


# Create and show dataframe for crime rates by zipcode and then as percentages
zip_codes = df.zip_code.value_counts().head(25)
display(zip_codes)
print('----------------------------------')
display(df.zip_code.value_counts(normalize=True).head(25))


# Visualizing the top 25 areas for crime 
df.zip_code.value_counts().head(25).plot.bar(rot=60, 
                                             title='Top 25 Zipcodes (2003-Present)')
plt.show()


# Out of all the areas in Austin, 78741 has the highest percentage of overall crime at 9.14%. This is a significant 1.29 percentage points higher than the number 2 area 78753 which hosts 7.85% of overall crime.

# #### Taking a closer look at particular areas... 
# 
# Because 78753 is my resident zipcode, I chose to examine it first. 
# 
# Next, I'll examine 78741. 

# <a id='q2'></a>
# ### B. Question 2. How is crime distributed in 78753? 

# In[8]:


# Examining crime in the 78753 area
df_53 = df[df.zip_code == 78753]

# Create a dataframe for the top 10 crime categories in the zipcode
df_53_off = df_53.highest_offense_description.value_counts().head(22)

# Display the different crime values & then as percentages 
display(df_53_off)
print('----------------------------------')
display(df_53.highest_offense_description.value_counts(normalize=True).head(22))

df_53_off.plot.pie(figsize=(8,8), 
                   title ='Crime Distribution (78753)')


# <a id='q3'></a>
# ### C. Question 3. How is crime distributed in 78741? 

# In[9]:


# Create a dataframe for crime in the 78741 area (the highest amount of crime of any Austin zip code)
df_41 = df[df.zip_code == 78741]

# Create a dataframe for the top 10 crime categories in the zipcode
df_41_off = df_41.highest_offense_description.value_counts().head(21)

# Display the different crime values & then as percentages 
display(df_41_off)
print('----------------------------------')
display(df_41.highest_offense_description.value_counts(normalize=True).head(21))

df_41_off.plot.pie(figsize=(8,8), 
                   title ='Crime Distribution (78741)')


# <a id='q4'></a>
# ### D. Question 4. How are violent crimes, in particular murder, capital murder, aggrivated assault, and rape distributed? 

# ***The following line of code shows crime rates only >= 1% per zipcode.***

# In[36]:


# Creating an overall and separate dataframes for violent crime
df_viol = df.query('highest_offense_description     == ["MURDER", "CAPITAL MURDER", "RAPE", "AGG ASSAULT"]') 
df_viol_mur = df.query('highest_offense_description == ["MURDER", "CAPITAL MURDER"]')
df_mur = df[df.highest_offense_description          == 'MURDER']
df_mur_cap = df[df.highest_offense_description      == 'CAPITAL MURDER']
df_agg_asslt = df[df.highest_offense_description    == 'AGG ASSAULT']
df_rape = df[df.highest_offense_description         == 'RAPE']

# Visualizing violent crimes per year
viol_per_year = df_viol['year'].value_counts().sort_index()

viol_per_year.plot.line(rot=60,
                        title='Annual Violent Crime Rates', 
                        fontsize=12)
plt.show()

# Visualizing murders per year
viol_mur_per_year = df_viol_mur['year'].value_counts().sort_index()

viol_mur_per_year.plot.line(rot=60, 
                            title='Annual Murder Rates', 
                            fontsize=12)
plt.show()

#Violent Crime by Zipcode
df_viol_zip = df_viol.zip_code.value_counts(normalize =True).head(25)
display(df_viol_zip)
df_viol_zip.plot.bar(title='Top Zipcodes for Violent Crime', 
                     fontsize=12,  
                     rot=60)
plt.show()

# Murder by Zipcode
display (df_viol_mur.zip_code.value_counts(normalize =True).head(25))
df_viol_mur.zip_code.value_counts().head(25).plot.bar(fontsize=12, 
                                                      title='Top Zipcodes for Murder', 
                                                      rot=60)
plt.show()
        
mur_by_hour = df_viol_mur['hour'].value_counts().sort_index()

# Visualizing hourly murder rate with Seaborn
f = sns.barplot(x=mur_by_hour.index, 
                y=mur_by_hour.values)
f.set_xticklabels(f.get_xticklabels(), 
                  rotation=60)
f.set(xlabel='Hour', 
      ylabel='Crimes Reported', 
      title ='Hourly Murder Rates')
plt.show()

# Calculating and visualizing frequency rate of violent crimes by zipcode
viol_freq = pd.crosstab(df_viol.zip_code, 
                        df_viol.highest_offense_description)

display(viol_freq)

viol_freq.plot.bar(figsize=(20,10), 
                   title='Violent Crime Distribution by Zipcode and Type since 2003', 
                   fontsize=12, 
                   stacked=True, 
                   rot=60)
plt.show()

viol_mur_freq = pd.crosstab(df_viol_mur.zip_code, 
                            df_viol_mur.highest_offense_description)

viol_mur_freq.plot.bar(figsize=(20,10), 
                       title='Murder Distribution by Zipcode and Type since 2003', 
                       fontsize=12, 
                       stacked=True,  
                       rot=60)
plt.show()


# According to the data , 2010 and 2016 had the most number of murders . Alarmingly, as of 10/19/2020, murders already totaled 34--the same amount for 2016 and 2010!!

# <a id='q5'></a>
# ### E. Question 5. What significance has the family violence factor played over time? 

# In[11]:


# Taking a look at first at the overall crime set
display(df.family_violence.mean())

print('----------------------------------')
display(df.groupby(df.index.year).family_violence.mean())

hrly_fam_viol_occurrences = df.groupby(df.index.year).family_violence.mean()

fam_viol_avg = df.groupby(df.index.year).family_violence.mean()

fam_viol_avg.plot(rot=60, 
                  title='Overall Family Violence Percentages (2003-Present)')
plt.show()

# Now taking a look at violent crime specifically 
display(df_viol.family_violence.mean())

print('----------------------------------')
display(df_viol.groupby(df_viol.index.year).family_violence.mean())

viol_hrly_fam_viol_occurrences = df_viol.groupby(df_viol.index.year).family_violence.mean()

viol_fam_viol_avg = df_viol.groupby(df_viol.index.year).family_violence.mean()

viol_fam_viol_avg.plot(rot=60, 
                       title='Violent Crime and Family Violence (2003-Present)')
plt.show()

# Now taking a look at murder with the family violence factor included 
display(df_viol_mur.family_violence.mean())

print('----------------------------------')
display(df_viol_mur.groupby(df_viol_mur.index.year).family_violence.mean())

mur_hrly_fam_viol_occurrences = df_viol_mur.groupby(df_viol_mur.index.year).family_violence.mean()

mur_fam_viol_avg = df_viol_mur.groupby(df_viol_mur.index.year).family_violence.mean()

mur_fam_viol_avg.plot(rot=60, 
                      title='Murder and Family Violence (2003-Present)')
plt.show()

# Now taking a look at rape with the family violence factor included 
display(df_rape.family_violence.mean())

print('----------------------------------')
display(df_rape.groupby(df_rape.index.year).family_violence.mean())

rape_hrly_fam_viol_occurrences = df_rape.groupby(df_rape.index.year).family_violence.mean()

rape_fam_viol_avg = df_rape.groupby(df_rape.index.year).family_violence.mean()

rape_fam_viol_avg.plot(rot=60, 
                       title='Rape and Family Violence(2003-Present)')
plt.show()

# Now taking a look at aggrivated assault with the family violence factor included 
display(df_rape.family_violence.mean())

print('----------------------------------')
display(df_agg_asslt.groupby(df_agg_asslt.index.year).family_violence.mean())

agg_asslt_fam_viol_avg = df_agg_asslt.groupby(df_agg_asslt.index.year).family_violence.mean()

agg_asslt_fam_viol_avg.plot(rot=60, 
                            title='Aggrivated Assault and Family Violence (2003-Present)')
plt.show()


# Overall, family violence is seeing an upward trend as a crime factor. Violent crime saw an alarming upward trend of the family violence factor, as well. Rapes, for example, involved the family violence factor a 3rd of the time in 2016 whereas in 2004, family violence was involved less than 1% of the time. 

# <a id='q6'></a>
# ### F. Question 6. How does murder appear on the map? 

# In[12]:


# As a heatmap
mur_coords = df_viol_mur[(df_viol_mur['latitude'].isnull() == False) 
                         & (df_viol_mur['longitude'].isnull() == False)]

k = folium.Map(location=[30.285516,-97.736753], 
               tiles='OpenStreetMap', 
               zoom_start=11) 
                         
k.add_child(plugins.HeatMap(mur_coords[['latitude', 
                                        'longitude']].values, 
                            radius=15))

k.save(outfile='aus_mur_heatmap.html')

k


# In[13]:


# Pinpointing individual addresses
df_viol_mur.dropna(subset=['latitude', 'longitude'], 
                   inplace=True)

# Making a folium map using incident lat and lon
m = folium.Map([30.2672, -97.7431], 
               tiles='Stamen Toner', 
               zoom_level=12)

for index, row in df_viol_mur.iterrows():
	lat = row['latitude']
	lon = row['longitude']
	name= row['address']
	folium.Marker([lat, lon], 
                  popup=name).add_to(m)
    
m.save(outfile='aus_mur_map.html')

m


# In[14]:


display(df.apd_sector.value_counts())

display(df.council_district.value_counts())

pd.crosstab(df.council_district, 
            df.apd_sector).plot.bar(stacked =True, 
                                    figsize =(12,8), 
                                    title   ='Incidents per Council Districts by APD Sector')


# ## IV. Summary
# Needless to say, violent crimes go hand-in-hand with other violent crimes.
# 
# So far, 78753 and 78741 are the top hotspots for all sorts of crime in Austin, including violent crime. 78753 accounts for 10.85% while  78741 accounts for 10.64% of total murders.
# 
# #### ***It is important to note that murder does not necessarily make the defendant(s) automatically eligible for the death penalty. Under Texas law, we distinguish capital murder, through the motives and actions of the defendant(s) during the commission of a homicide, as whether or not automatically warranting an eventual date with the executioner. This includes such things as if the homicide was premeditated or not, if the defendant(s) murdered a police officer, etc.***
# 
# 78723 comes in at number one with 14.1% of total capital murders. 
# 
# Next, 78741 climbs back to claim the number 1 spot for rape at 12.09% -- 3.43 percentage points higher than the number 2 spot 78753 carrying 8.66% which is quite a significant lead when you look at it on the graph!! Why does rape occur so much more often in this area than in others?
# 
# A peculiar outlier is zipcode 78731. Although violent crime frequency ranks amongst the lowest there, rape accounts for over 50% of violent crimes committed in that area. Why is that? 
# 
# Astonishingly the family violence factor played an ever increasing role over over time, in regards to violent crime. From 2003 to 2015, family violence increased by nearly 10 percentage points--meaning you were likely to be the victim of a family member, during the commission of a rape, aggrivated assault, murder, or capital murder, only 3.15% of the time in 2003. But by 2015, that same likelihood rose to 12.82%!
