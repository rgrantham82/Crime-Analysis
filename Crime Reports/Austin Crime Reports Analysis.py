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
#     IV. Time Series Modeling with Facebook Prophet 
#     
#     Questions:
# ><ul>
# ><li><a href="#q1"> 1. What areas of Austin have the highest crime rates?</a></li>
# ><li><a href="#q2"> 2. How is crime distributed in 78701?</a></li> 
# ><li><a href="#q3"> 3. How is crime distributed in 78753?</a></li>     
# ><li><a href="#q4"> 4. How is crime distributed in 78741?</a></li>
# ><li><a href="#q5"> 5. How is crime distributed in 78745?</a></li>
# ><li><a href="#q6"> 6. How are violent crimes, in particular murder, capital murder, aggrivated assault, and rape distributed?
# ><li><a href="#q7"> 7. How is crime distributed across council districts?
# ><li><a href="#q8"> 8. How does murder appear on the map?
# ><li><a href="#q9"> 9. Are there any addresses where murder occurs frequently?
# </a></li>

# ## I. Introduction
# 
# I began reviewing the Crime Reports dataset, provided by the Austin PD, around the same time I began reviewing its Hate Crimes datasets for analysis, at the beginning of 2020. This is a rather large dataset, containing over 2 million records, spanning from 2003 to the present, and is updated weekly. 
# 
# This is a self-paced project, conceived outside of both work and the educational arenas. It is my hope that this project will reveal some actionable insights that will benefit the Austin law enforcement community, news outlets, and anyone else interested in gaining knowledge on how best to combat the problem of crime in the Austin area.
# 
# I first attempted importing the data into this notebook using Sodapy's Socrata API method but found it lacking. It didn't import the entire dataset, and added several redundant columns. I, therefore, prefer to manually download the entire dataset and re-download each week after it's updated.

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import plugins
import seaborn as sns
import warnings
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("seaborn-white")


# In[2]:


df = pd.read_csv("crime_reports.csv")


# In[3]:


display(df.info())
display(df.head())
display(df.tail())


# ## II. Data Scrubbing
# 
# There are several columns of data we don't need. We'll drop those and also scrub the Columns were keeping for analysis. Mainly, we want the zip code and address columns to be free of nulls and duplicates. We'll also create new columns for time series analysis. 

# In[4]:


def clean_data(df):
    drop_col = [
        "Occurred Time",
        "Occurred Date",
        "Highest Offense Code",
        "Census Tract",
        "Family Violence",
        "Clearance Status",
        "Report Date",
        "Report Time",
        "Clearance Date",
        "UCR Category",
        "Category Description",
        "X-coordinate",
        "Y-coordinate",
        "Location",
    ]
    clean_col = ["Occurred Date Time", "Report Date Time"]
    df.drop(drop_col, axis=1, inplace=True)
    df.dropna(subset=clean_col, inplace=True)
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    date_col = ["occurred_date_time", "report_date_time"]
    cat_col = ["highest_offense_description", "location_type", "apd_sector"]
    df[date_col] = df[date_col].astype("datetime64")
    df[cat_col] = df[cat_col].astype("category")
    df["year"] = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.year
    df["month"] = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.month
    df["week"] = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.week
    df["day"] = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.day
    df["hour"] = pd.to_datetime(df["occurred_date_time"], format="%m/%d/%Y").dt.hour
    df.set_index(["occurred_date_time"], inplace=True)
    df.sort_index(inplace=True)
    return df


df = clean_data(df)


# In[29]:


display(df.info())
display(df.head())
display(df.tail())


# ## III. Exploratory Analysis

# First, let's get an overall look at crime rates and how they trend over time...

# #### Overall crime rates over time 

# In[30]:


# plotting trend on a monthly basis
plt.figure(figsize=(10, 5))
plt.plot(df.resample("M").size())
plt.title("Monthly trend, 2003-Present")
plt.show()


# Above plot re-shown as rolling average
plt.figure(figsize=(10, 5))
df.resample("D").size().rolling(365).sum().plot()
plt.title("365 day rolling average, 2003-Present")
plt.show()


# Creating and visualizing a data frame for the overall yearly crime rate since 2003
crimes_per_year = df["year"].value_counts().sort_index()
g = sns.barplot(x=crimes_per_year.index, y=crimes_per_year.values)
g.set_xticklabels(g.get_xticklabels(), rotation=60)
g.set(xlabel="Year", ylabel="Crimes Reported", title="Annual Crime Rates")
plt.show()


# Creating and visualizing a data frame for the overall yearly crime rate since 2003
crimes_per_month = df["month"].value_counts().sort_index()
d = sns.barplot(x=crimes_per_month.index, y=crimes_per_month.values)
d.set_xticklabels(d.get_xticklabels(), rotation=60)
d.set(xlabel="Month", ylabel="Crimes Reported", title="Monthly Crime Rates")
plt.show()


# Overall hourly crime rates as well
crimes_per_hour = df["hour"].value_counts().sort_index()
e = sns.barplot(x=crimes_per_hour.index, y=crimes_per_hour.values)
e.set_xticklabels(e.get_xticklabels(), rotation=60)
e.set(xlabel="Hour", ylabel="Crimes Reported", title="Hourly Crime Rates")
plt.show()


# #### Top 25 crime types 

# In[31]:


df.highest_offense_description.value_counts().head(25).sort_values().plot.barh(
    figsize=(8, 6), title="Top 25 crime types (2003-Present)"
)


# Between 2003 and now, crime peaked in 2008 and continued a downward trend until 2019 when it rose again. Since we're still in 2020, we have to wait until the end of the year to see what 2020 yields. 

# <a id='q1'></a>
# ### A. Question 1. What areas of Austin have the highest crime rates? 

# In[32]:


# Create and show dataframe for crime rates by zipcode and then as percentages
zip_codes = df.zip_code.value_counts().head(25)
display(zip_codes)
print("----------------------------------")
display(df.zip_code.value_counts(normalize=True).head(25))


# Visualizing the top 25 areas for crime
df.zip_code.value_counts().head(25).plot.bar(
    rot=60, title="Top 25 zip codes, overall crime"
)
plt.show()


# Out of all the areas in Austin, 78741 has the highest percentage of overall crime at 9.05%. This is a significant 1.23 percentage points higher than the number 2 area 78753 which hosts 7.82% of overall crime.

# #### Taking a closer look at particular areas... 
# 
# The next section will examine the zip codes 78701 (downtown), 78753, 78741, and 78745.

# <a id='q2'></a>
# ### B. Question 2. How is crime distributed in 78701? 

# In[33]:


# Examining crime in the 78701 area
df_01 = df[df.zip_code == 78701]


# Create a dataframe for the top crime categories in the zipcode
df_01_off = df_01.highest_offense_description.value_counts().head(24)


# Display the different crime values & then as percentages
display(df_01_off)
print("----------------------------------")
display(df_01.highest_offense_description.value_counts(normalize=True).head(24))
df_01_off.plot.pie(figsize=(8, 8), title="Crime Distribution (78701)")


# <a id='q3'></a>
# ### C. Question 2. How is crime distributed in 78753? 

# In[34]:


# Examining crime in the 78753 area
df_53 = df[df.zip_code == 78753]


# Create a dataframe for the top 10 crime categories in the zipcode
df_53_off = df_53.highest_offense_description.value_counts().head(22)


# Display the different crime values & then as percentages
display(df_53_off)
print("----------------------------------")
display(df_53.highest_offense_description.value_counts(normalize=True).head(22))
df_53_off.plot.pie(figsize=(8, 8), title="Crime Distribution (78753)")


# <a id='q4'></a>
# ### D. Question 3. How is crime distributed in 78741? 

# In[35]:


# Examining crime in the 78741 area (the highest amount of crime of any Austin zip code)
df_41 = df[df.zip_code == 78741]


# Create a dataframe for the top crime categories in the zipcode
df_41_off = df_41.highest_offense_description.value_counts().head(21)


# Display the different crime values & then as percentages
display(df_41_off)
print("----------------------------------")
display(df_41.highest_offense_description.value_counts(normalize=True).head(21))
df_41_off.plot.pie(figsize=(8, 8), title="Crime Distribution (78741)")


# <a id='q5'></a>
# ### E. Question 4. How is crime distributed in 78745?

# In[36]:


# Examining crime in the 78745 area
df_45 = df[df.zip_code == 78745]


# Create a dataframe for the top 10 crime categories in the zipcode
df_45_off = df_45.highest_offense_description.value_counts().head(22)


# Display the different crime values & then as percentages
display(df_45_off)
print("----------------------------------")
display(df_45.highest_offense_description.value_counts(normalize=True).head(22))
df_45_off.plot.pie(figsize=(8, 8), title="Crime Distribution (78745)")


# <a id='q6'></a>
# ### F. Question 5. How are violent crimes, in particular murder, capital murder, aggrivated assault, and rape distributed? 

# In[37]:


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
viol_per_year.plot.bar(rot=60, title="Annual Violent Crime Rates")
plt.show()

# As rolling average
df_viol.resample("D").size().rolling(365).sum().plot(
    figsize=(10, 5), title="365 day rolling average for violent crime"
)
plt.show()


# Visualizing murders per year
viol_mur_per_year = df_viol_mur.year.value_counts().sort_index()
viol_mur_per_year.plot.bar(rot=60, title="Annual Murder Rates")
plt.show()

# As rolling average
df_viol_mur.resample("D").size().rolling(365).sum().plot(
    figsize=(10, 5), title="365 day rolling average for murders"
)
plt.show()


# Overall violent crime by Zipcode
display(df_viol.zip_code.value_counts(normalize=True).head(25))
df_viol.zip_code.value_counts().head(25).plot.bar(
    title="Top 25 Zipcodes for Violent Crime", rot=60
)
plt.show()


# Calculating and visualizing frequency rate of violent crimes types by zipcode
viol_freq = pd.crosstab(df_viol.zip_code, df_viol.highest_offense_description)
display(viol_freq)
viol_freq.plot.barh(
    title="Violent Crime Distribution by Zipcode and Type since 2003",
    stacked=True,
    figsize=(8, 11),
)
plt.show()

# Overall murders by zip code
mur_freq = pd.crosstab(df_viol_mur.zip_code, df_viol_mur.highest_offense_description)
mur_freq.plot.barh(
    figsize=(8, 8),
    stacked=True,
    title="Murder Distribution by Zipcode and Type since 2003",
)
plt.show()

display(df_viol_mur.zip_code.value_counts(normalize=True).head(25))
df_viol_mur.zip_code.value_counts().head(25).plot.bar(
    title="Top 25 Zipcodes for Murder", rot=60
)
plt.show()

# Visualizing monthly & hourly murder rates
mur_by_month = df_viol_mur["month"].value_counts().sort_index()
mur_by_hour = df_viol_mur["hour"].value_counts().sort_index()

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


# According to the data , 2010 and 2016 had the most number of murders . Alarmingly, as of 11/23/2020, we've now had more murders this year than any other since 2003.
# 
# So, you're most likely to get murdered in July, between 1 and 2am, in the 78753 zip code, with 78741 coming in as a very strong alternate. Good to know!

# <a id='q7'></a>
# ### G. Question 7. How is crime distributed across council districts and APD sectors?
# 
# #### checking council districts, APD districts, and sectors for overall crime rates 

# In[51]:


df.council_district.value_counts().plot.bar(
    title="Council districts, overall crime", rot=60
)
plt.show()

df.apd_sector.value_counts().plot.bar(
    title="APD sectors, overall crime", fontsize=12, figsize=(14, 7), rot=60
)
plt.show()

df.apd_district.value_counts().plot.bar(
    title="APD districts, overall crime", rot=60, fontsize=12, figsize=(14, 7)
)
plt.show()


# #### Distribution of violent crime and murders across council districts and APD sectors 

# In[68]:


pd.crosstab(df_viol.council_district, df_viol.highest_offense_description).plot.bar(
    stacked=True,
    figsize=(12, 6),
    rot=60,
    title="Violent crime distribution by council district",
)
plt.show()


pd.crosstab(
    df_viol_mur.council_district, df_viol_mur.highest_offense_description
).plot.bar(
    figsize=(12, 6),
    rot=60,
    title="Murder distribution by council district",
)
plt.show()


pd.crosstab(df_viol.apd_sector, df_viol.highest_offense_description).plot.bar(
    figsize=(12, 6),
    stacked=True,
    rot=60,
    title="Violent crime distribution by APD sector",
)
plt.show()


pd.crosstab(df_viol_mur.apd_sector, df_viol_mur.highest_offense_description).plot.bar(
    figsize=(12, 6), rot=60, title="Murder distribution by APD sector"
)
plt.show()


pd.crosstab(df_viol.apd_district, df_viol.highest_offense_description).plot.bar(
    figsize=(12, 6),
    stacked=True,
    rot=60,
    title="Violent crime distribution by APD district",
)
plt.show()

pd.crosstab(df_viol_mur.apd_district, df_viol_mur.highest_offense_description).plot.bar(
    figsize=(12, 6), rot=60, title="Murder distribution by APD district"
)
plt.show()


# #### Violent crime and murder distribution by location type

# In[77]:


viol_loc = pd.crosstab(df_viol.location_type, df_viol.highest_offense_description)
display(viol_loc)

mur_loc = pd.crosstab(
    df_viol_mur.location_type, df_viol_mur.highest_offense_description
)

viol_loc.plot.barh(
    figsize=(8, 12),
    stacked=True,
    title="Violent crime distribution by location type since 2003",
)
plt.show()

mur_loc.plot.barh(
    figsize=(8, 8), title="Murder distribution by location type since 2003"
)
plt.show()


# <a id='q8'></a>
# ### H. Question 8. How does murder appear on the map? 

# In[46]:


# As a heatmap

mur_coords_heat = df_viol_mur[
    (df_viol_mur["latitude"].isnull() == False)
    & (df_viol_mur["longitude"].isnull() == False)
]

k = folium.Map(location=[30.2672, -97.7431], tiles="OpenStreetMap", zoom_start=12)

k.add_child(
    plugins.HeatMap(mur_coords_heat[["latitude", "longitude"]].values, radius=15)
)

k.save(outfile="aus_mur_heatmap.html")

k


# In[17]:


# Pinpointing individual addresses

mur_coords_add = df_viol_mur[
    (df_viol_mur["latitude"].isnull() == False)
    & (df_viol_mur["longitude"].isnull() == False)
]

m = folium.Map([30.2672, -97.7431], tiles="OpenStreetMap", zoom_level=12)

for (index, row) in mur_coords_add.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]
    name = row["address"]
    folium.Marker([lat, lon], popup=name).add_to(m)

m.save(outfile="aus_mur_map.html")

m


# <a id='q9'></a>
# ### I. Question 9. Are there any addresses where murder occurs frequently?

# In[18]:


df_viol_mur.address.value_counts().head(31)


# ## IV. Prediction Modeling 

# ### A. Time Series Modeling of the overall dataframe with Facebook Prophet.

# In[19]:


df_fbprophet = df

df_m_1 = df_fbprophet.resample("W").size().reset_index()
df_m_1.columns = ["date", "weekly_crime_count"]
df_m_final_1 = df_m_1.rename(columns={"date": "ds", "weekly_crime_count": "y"})

m_1 = Prophet(interval_width=0.95, yearly_seasonality=False)
m_1.add_seasonality(name="monthly", period=30.5, fourier_order=10)
m_1.add_seasonality(name="quarterly", period=91.5, fourier_order=10)
m_1.add_seasonality(name="weekly", period=52, fourier_order=10)
m_1.fit(df_m_final_1)

future_1 = m_1.make_future_dataframe(periods=104, freq="W")

pred_1 = m_1.predict(future_1)

fig2_1 = m_1.plot_components(pred_1)
fig2_2 = plot_plotly(m_1, pred_1)
fig2_2


# #### ...now the violent crime dataframe

# In[20]:


df_viol_fbprophet = df_viol

df_v = df_viol_fbprophet.resample("W").size().reset_index()
df_v.columns = ["date", "weekly_crime_count"]
df_v_final = df_v.rename(columns={"date": "ds", "weekly_crime_count": "y"})

v = Prophet(interval_width=0.95, yearly_seasonality=False)
v.add_seasonality(name="monthly", period=30.5, fourier_order=10)
v.add_seasonality(name="quarterly", period=91.5, fourier_order=10)
v.add_seasonality(name="weekly", period=52, fourier_order=10)
v.fit(df_v_final)

future = v.make_future_dataframe(periods=104, freq="W")
pred = v.predict(future)
fig2_1 = v.plot_components(pred)
fig2_3 = plot_plotly(v, pred)
fig2_3


# #### ...now the murder dataframe 

# In[21]:


df_viol_mur_fbprophet = df_viol_mur

df_m = df_viol_mur_fbprophet.resample("W").size().reset_index()
df_m.columns = ["date", "weekly_crime_count"]
df_m_final = df_m.rename(columns={"date": "ds", "weekly_crime_count": "y"})

m = Prophet(interval_width=0.95, yearly_seasonality=False)
m.add_seasonality(name="monthly", period=30.5, fourier_order=10)
m.add_seasonality(name="quarterly", period=91.5, fourier_order=10)
m.add_seasonality(name="weekly", period=52, fourier_order=10)
m.fit(df_m_final)

future = m.make_future_dataframe(periods=104, freq="W")

pred = m.predict(future)
fig3_1 = m.plot_components(pred)
fig3_3 = plot_plotly(m, pred)
fig3_3


# #### ...now examining some zip codes
# 
# #### 78701

# In[22]:


df_fbprophet_01 = df_01

df_m_01 = df_fbprophet_01.resample("W").size().reset_index()
df_m_01.columns = ["date", "weekly_crime_count"]
df_m_final_01 = df_m_01.rename(columns={"date": "ds", "weekly_crime_count": "y"})

m_01 = Prophet(interval_width=0.95, yearly_seasonality=False)
m_01.add_seasonality(name="monthly", period=30.5, fourier_order=10)
m_01.add_seasonality(name="quarterly", period=91.5, fourier_order=10)
m_01.add_seasonality(name="weekly", period=52, fourier_order=10)
m_01.fit(df_m_final_01)

future_01 = m_01.make_future_dataframe(periods=104, freq="W")
pred_01 = m_01.predict(future)
fig2_01 = m_01.plot_components(pred)
fig2_01_1 = plot_plotly(m_01, pred_01)
fig2_01_1


# #### 78753

# In[23]:


df_fbprophet_53 = df_53

df_m_53 = df_fbprophet_53.resample("W").size().reset_index()
df_m_53.columns = ["date", "weekly_crime_count"]
df_m_final_53 = df_m_53.rename(columns={"date": "ds", "weekly_crime_count": "y"})

m_53 = Prophet(interval_width=0.95, yearly_seasonality=False)
m_53.add_seasonality(name="monthly", period=30.5, fourier_order=10)
m_53.add_seasonality(name="quarterly", period=91.5, fourier_order=10)
m_53.add_seasonality(name="weekly", period=52, fourier_order=10)
m_53.fit(df_m_final_53)

future_53 = m_53.make_future_dataframe(periods=104, freq="W")
pred_53 = m_53.predict(future)
fig2_53 = m_53.plot_components(pred)
fig2_53_1 = plot_plotly(m_53, pred_53)
fig2_53_1


# #### 78741

# In[24]:


df_fbprophet_41 = df_41

df_m_41 = df_fbprophet_41.resample("W").size().reset_index()
df_m_41.columns = ["date", "weekly_crime_count"]
df_m_final_41 = df_m_41.rename(columns={"date": "ds", "weekly_crime_count": "y"})

m_41 = Prophet(interval_width=0.95, yearly_seasonality=False)
m_41.add_seasonality(name="monthly", period=30.5, fourier_order=10)
m_41.add_seasonality(name="quarterly", period=91.5, fourier_order=10)
m_41.add_seasonality(name="weekly", period=52, fourier_order=10)
m_41.fit(df_m_final_41)

future_41 = m_41.make_future_dataframe(periods=104, freq="W")
pred_41 = m_41.predict(future)
fig2_41 = m_41.plot_components(pred)
fig2_41_1 = plot_plotly(m_41, pred_53)
fig2_41_1


# #### 78745

# In[25]:


df_fbprophet_45 = df_45

df_m_45 = df_fbprophet_45.resample("W").size().reset_index()
df_m_45.columns = ["date", "weekly_crime_count"]
df_m_final_45 = df_m_45.rename(columns={"date": "ds", "weekly_crime_count": "y"})

m_45 = Prophet(interval_width=0.95, yearly_seasonality=False)
m_45.add_seasonality(name="monthly", period=30.5, fourier_order=10)
m_45.add_seasonality(name="quarterly", period=91.5, fourier_order=10)
m_45.add_seasonality(name="weekly", period=52, fourier_order=10)
m_45.fit(df_m_final_45)

future_45 = m_45.make_future_dataframe(periods=104, freq="W")
pred_45 = m_45.predict(future)
fig2_45 = m_45.plot_components(pred)
fig2_45_1 = plot_plotly(m_45, pred_45)
fig2_45_1

