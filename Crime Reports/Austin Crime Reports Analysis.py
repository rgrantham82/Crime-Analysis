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
# I first attempted importing the data into this notebook using Sodapy's Socrata API method but found it lacking. It didn't import the entire dataset, and added several redundant columns. I, therefore, prefer to manually download the entire dataset and re-download each week after it's updated.

# In[1]:


# Importing essential libraries and configurations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins
import seaborn as sns
import warnings
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

plt.style.use("classic")
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loading the data
df = pd.read_csv('crime_reports.csv')


# In[3]:


# Examining the dataframe
display(df.info())
print('----------------------------------')
display(df.duplicated().sum())
print('----------------------------------')
display(df.isnull().sum())


# ## II. Data Scrubbing
# 
# There are several columns of data we don't need. We'll drop those and also scrub the Columns were keeping for analysis. Mainly, we want the zip code and address columns to be free of nulls and duplicates. 
# 
# The 'clearance status' column contains 3 types of statuses: Y for Yes, N for No, and O which stands for "cleared by other means than arrest." Therefore, I changed it to boolean type:  Y and O as True, and N as False. However, you may note that areas, where there is no clearance status at all, may or may not contain a corresponding date in the clearance date column. I am unsure how best to handle this so I am open to suggestions or advice. I also converted the 'family violence' column to boolean type.  

# In[4]:


# Data-scrubbing script
def clean_data(df):
    drop_col = [
        "Occurred Time",
        "Occurred Date",
        "Report Date",
        "Report Time",
        "Census Tract",
        "UCR Category",
        "Category Description",
        "X-coordinate",
        "Y-coordinate",
        "Location"
    ]
    clean_col = [
        "Report Date Time", 
        "Occurred Date Time", 
        "Zip Code"
    ]
    df.drop(drop_col, axis=1, inplace=True)
    df.dropna(subset=clean_col, inplace=True)
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    """Convert the following to bools"""
    d = {"Y": True, "N": False}
    e = {"C": True, "O": True, "N": False}
    df.clearance_status = df.clearance_status.map(e)
    df.clearance_status = df.clearance_status.astype("bool")
    df.family_violence  = df.family_violence.map(d)
    df.family_violence  = df.family_violence.astype("bool")
    """Convert the following to datetime type"""
    date_col = [
        "occurred_date_time", 
        "clearance_date", 
        "report_date_time"
    ]
    """Convert the following to integer type"""
    int_col  = ["zip_code"]
    """Convert the following to category type"""
    cat_col  = [
        "highest_offense_description", 
        "location_type", 
        "apd_sector"
    ]
    df[date_col] = df[date_col].astype("datetime64")
    df[cat_col]  = df[cat_col].astype("category")
    df[int_col]  = df[int_col].astype("int64")
    """Creating new time columns and an index out of the 'occurred date time' column"""
    df["year"]   = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.year
    df["month"]  = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.month
    df["week"]   = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.week
    df["day"]    = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.day
    df["hour"]   = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.hour
    df.set_index(["occurred_date_time"], inplace=True)
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


# In[6]:


# Plotting yearly and weekly trends
plt.figure(figsize=(10,5))
plt.plot(df.resample('Y').size())
plt.xlabel('Yearly')
plt.ylabel('Number of crimes')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df.resample('W').size())
plt.xlabel('Weekly')
plt.ylabel('Number of crimes')
plt.show()


# ## III. Exploratory Analysis

# First, let's get an overall look at crime rates and how they trend over time...

# ### Overall crime rates over time 

# In[7]:


figsize=(20,10)

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

# Creating and visualizing a data frame for the overall yearly crime rate since 2003
crimes_per_month = df['month'].value_counts().sort_index() 

d = sns.barplot(x=crimes_per_month.index, 
                y=crimes_per_month.values)
d.set_xticklabels(d.get_xticklabels(), 
                  rotation=60)
d.set(xlabel='Month', 
      ylabel='Crimes Reported', 
      title ='Monthly Crime Rates')
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


# ### Top 50 crime types 

# In[8]:


df.highest_offense_description.value_counts().head(50).sort_values().plot.barh(
    figsize=(9, 10), title="Top 50 crime types since 2003"
)


# Between 2003 and now, crime peaked in 2008 and continued a downward trend until 2019 when it rose again. Since we're still in 2020, we have to wait until the end of the year to see what 2020 yields. 

# <a id='q1'></a>
# ### A. Question 1. What areas of Austin have the highest crime rates? 

# In[9]:


# Create and show dataframe for crime rates by zipcode and then as percentages
zip_codes = df.zip_code.value_counts().head(25)
display(zip_codes)
print("----------------------------------")
display(df.zip_code.value_counts(normalize=True).head(25))


# Visualizing the top 25 areas for crime 
df.zip_code.value_counts().head(25).plot.bar(
    rot=60, title="Top 25 Zipcodes (2003-Present)"
)
plt.show()


# Out of all the areas in Austin, 78741 has the highest percentage of overall crime at 9.14%. This is a significant 1.29 percentage points higher than the number 2 area 78753 which hosts 7.85% of overall crime.

# #### Taking a closer look at particular areas... 
# 
# Because 78753 is my resident zipcode, I chose to examine it first. 
# 
# Next, I'll examine 78741. 

# <a id='q2'></a>
# ### B. Question 2. How is crime distributed in 78753? 

# In[10]:


# Examining crime in the 78753 area
df_53 = df[df.zip_code == 78753]

# Create a dataframe for the top 10 crime categories in the zipcode
df_53_off = df_53.highest_offense_description.value_counts().head(22)

# Display the different crime values & then as percentages 
display(df_53_off)
print("----------------------------------")
display(df_53.highest_offense_description.value_counts(normalize=True).head(22))

df_53_off.plot.pie(figsize=(8, 8), title="Crime Distribution (78753)")


# <a id='q3'></a>
# ### C. Question 3. How is crime distributed in 78741? 

# In[11]:


# Create a dataframe for crime in the 78741 area (the highest amount of crime of any Austin zip code)
df_41 = df[df.zip_code == 78741]

# Create a dataframe for the top 10 crime categories in the zipcode
df_41_off = df_41.highest_offense_description.value_counts().head(21)

# Display the different crime values & then as percentages 
display(df_41_off)
print("----------------------------------")
display(df_41.highest_offense_description.value_counts(normalize=True).head(21))

df_41_off.plot.pie(figsize=(8, 8), title="Crime Distribution (78741)")


# <a id='q4'></a>
# ### D. Question 4. How are violent crimes, in particular murder, capital murder, aggrivated assault, and rape distributed? 

# In[12]:


# Creating an overall and separate dataframes for violent crime
df_viol = df.query(
    'highest_offense_description == ["MURDER", "CAPITAL MURDER", "RAPE", "AGG ASSAULT"]'
)
df_viol_mur = df.query('highest_offense_description == ["MURDER", "CAPITAL MURDER"]')
df_mur = df[df.highest_offense_description == "MURDER"]
df_mur_cap = df[df.highest_offense_description == "CAPITAL MURDER"]
df_agg_asslt = df[df.highest_offense_description == "AGG ASSAULT"]
df_rape = df[df.highest_offense_description == "RAPE"]


# Visualizing violent crimes per year
viol_per_year = df_viol["year"].value_counts().sort_index()
viol_per_year.plot.bar(
    rot=60, title="Annual Violent Crime Rates (2003-Present)", fontsize=12
)
plt.show()

# Visualizing murders per year
viol_mur_per_year = df_viol_mur.year.value_counts().sort_index()
viol_mur_per_year.plot.bar(
    rot=60, title="Annual Murder Rates (2003-Present)", fontsize=12
)
plt.show()

# Violent Crime by Zipcode
display(df_viol.zip_code.value_counts(normalize=True).head(25))
df_viol.zip_code.value_counts().head(25).plot.bar(
    title="Top Zipcodes for Violent Crime", fontsize=12, rot=60
)
plt.show()

# Murder by Zipcode
display(df_viol_mur.zip_code.value_counts(normalize=True).head(25))
df_viol_mur.zip_code.value_counts().head(25).plot.bar(
    fontsize=12, title="Top Zipcodes for Murder", rot=60
)
plt.show()

mur_by_month = df_viol_mur["month"].value_counts().sort_index()
mur_by_hour = df_viol_mur["hour"].value_counts().sort_index()

# Visualizing monthly & hourly murder rate with Seaborn
v = sns.barplot(x=mur_by_month.index, y=mur_by_month.values)
v.set_xticklabels(v.get_xticklabels(), rotation=60)
v.set(
    xlabel="Month",
    ylabel="Crimes Reported",
    title="Monthly Murder Rates (2003-Present)",
)
plt.show()

f = sns.barplot(x=mur_by_hour.index, y=mur_by_hour.values)
f.set_xticklabels(f.get_xticklabels(), rotation=60)
f.set(
    xlabel="Hour", ylabel="Crimes Reported", title="Hourly Murder Rates (2003-Present)"
)
plt.show()

# Calculating and visualizing frequency rate of violent crimes by zipcode
viol_freq = pd.crosstab(df_viol.zip_code, df_viol.highest_offense_description)

display(viol_freq)

viol_freq.plot.bar(
    figsize=figsize,
    title="Violent Crime Distribution by Zipcode and Type since 2003",
    fontsize=12,
    stacked=True,
    rot=60,
)
plt.show()

viol_mur_freq = pd.crosstab(
    df_viol_mur.zip_code, df_viol_mur.highest_offense_description
)

viol_mur_freq.plot.bar(
    figsize=figsize,
    title="Murder Distribution by Zipcode and Type since 2003",
    fontsize=12,
    stacked=True,
    rot=60,
)
plt.show()


# In[13]:


viol_freq.to_csv('viol_freq.csv')
df_viol.to_csv('df_viol.csv')
df_viol_mur.to_csv('df_viol_mur.csv')


# According to the data , 2010 and 2016 had the most number of murders . Alarmingly, as of 10/19/2020, murders already totaled 34--the same amount for 2016 and 2010!!
# 
# So, you're most likely to get murdered in July, between 1 and 2am, in the 78753 zip code, with 78741 coming in as a very strong alternate. Good to know!

# Overall, family violence is seeing an upward trend as a crime factor. Violent crime saw an alarming upward trend of the family violence factor, as well. Rapes, for example, involved the family violence factor a 3rd of the time in 2016 whereas in 2004, family violence was involved less than 1% of the time. 

# <a id='q6'></a>
# ### F. Question 6. How does murder appear on the map? 

# In[14]:


# As a heatmap
mur_coords_heat = df_viol_mur[(df_viol_mur['latitude'].isnull()    == False) 
                              & (df_viol_mur['longitude'].isnull() == False)]

k = folium.Map(location=[30.2672, -97.7431], tiles='OpenStreetMap', zoom_start=11) 
                         
k.add_child(plugins.HeatMap(mur_coords_heat[['latitude', 'longitude']].values, radius=15))

k.save(outfile='aus_mur_heatmap.html')

k


# In[15]:


# Pinpointing individual addresses
mur_coords_add  = df_viol_mur[(df_viol_mur['latitude'].isnull() == False) & (df_viol_mur['longitude'].isnull() == False)]

# Making a folium map using incident lat and lon
m = folium.Map([30.2672, -97.7431], tiles='OpenStreetMap', zoom_level=12)

for index, row in mur_coords_add.iterrows():
	lat = row['latitude']
	lon = row['longitude']
	name= row['address']
	folium.Marker([lat, lon], popup=name).add_to(m)
    
m.save(outfile='aus_mur_map.html')

m


# ## Are there any addresses where murder occurs frequently?

# In[16]:


df_viol_mur.address.value_counts().head(31)


# In[17]:


df2 = df.copy()
df2.to_csv('df2_crime.csv')


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

# ### Time Series Modeling of the murder data with Facebook Prophet 

# In[22]:


df_viol_mur_fbprophet = df_viol_mur[(df_viol_mur.year >= 2003) & (df_viol_mur.year < 2020)]

df_m         = df_viol_mur_fbprophet.resample('M').size().reset_index()
df_m.columns = ['date', 'monthly_crime_count']
df_m_final   = df_m.rename(columns = {'date': 'ds', 'monthly_crime_count': 'y'})

df_m_final.head(), df_m_final.tail()


# In[23]:


m = Prophet(interval_width=0.95, yearly_seasonality=False)
m.add_seasonality(name='monthly', period=30.5, fourier_order=10)
m.add_seasonality(name='quarterly', period=91.5, fourier_order=10)
m.add_seasonality(name='weekly', period=52.25, fourier_order=10)
m.fit(df_m_final)


# In[28]:


future = m.make_future_dataframe(periods=24, 
                                 freq='M')
pred   = m.predict(future)

pred.to_csv('pred.csv')


# In[26]:


fig2 = m.plot_components(pred)


# In[27]:


plot_plotly(m, pred)


# In[ ]:




