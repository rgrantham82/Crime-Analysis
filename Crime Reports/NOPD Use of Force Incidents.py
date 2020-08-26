#!/usr/bin/env python
# coding: utf-8

# # NOPD Use of Force Incidents

# In[1]:


# Importing essential libraries and configurations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

get_ipython().magic('matplotlib inline')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# In[2]:


plt.style.use('seaborn-white')


# In[3]:


# loading the data to remove duplicates initially 
df_with_duplicates = pd.read_csv(r'C:\Users\Robert\Downloads\NOPD_Use_of_Force_Incidents.csv')

df = df_with_duplicates.drop_duplicates()


# In[4]:


display(df.info())
display(df.isnull().sum())


# In[5]:


def clean_data(df):
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    """Cleansing the following columns of null values"""
    clean_col = ['division', 
                 'officer_age', 
                 'officer_years_of_service',
                 'subject_age', 
                 'subject_ethnicity', 
                 'use_of_force_reason']
    """Converting the following columns to categories"""
    cat_col = ['originating_bureau', 
               'division_level', 
               'division', 
               'unit', 
               'investigation_status', 
               'service_type', 
               'light_condition', 
               'weather_condition']
    """Converting the following columns from float to integer"""
    int_col = ['officer_years_of_service', 
               'subject_age', 
               'officer_age', ]
    df.dropna(subset=clean_col, 
              inplace=True)
    d = {'Yes': True, 'No': False}
    df[cat_col] = df[cat_col].astype('category')
    df[int_col] = df[int_col].astype('int64')
    df['use_of_force_effective'] = df['use_of_force_effective'].map(d)
    df['officer_injured'] = df['officer_injured'].map(d)
    df['subject_injured'] = df['subject_injured'].map(d)
    df['subject_hospitalized'] = df['subject_hospitalized'].map(d)
    df['subject_arrested'] = df['subject_arrested'].map(d)
    df['use_of_force_effective'] = df['use_of_force_effective'].astype('bool')
    df['officer_injured'] = df['officer_injured'].astype('bool')
    df['subject_injured'] = df['subject_injured'].astype('bool')
    df['subject_hospitalized'] = df['subject_hospitalized'].astype('bool')
    df['subject_arrested'] = df['subject_arrested'].astype('bool')
    """Converting and setting the index out of the 'date_occurred' column"""
    df['date_occurred'] = df['date_occurred'].astype('datetime64')
    df.set_index('date_occurred', 
                 inplace=True)
    df.sort_index(inplace=True)
    return df
df = clean_data(df)


# In[6]:


display(df.info())
display(df.isnull().sum())
display(df.head())
display(df.tail())


# ### Examining the officer according to a few variables... 

# In[7]:


print('------------------------------------------')
print('Average Age of Police Officers by Division')
print('------------------------------------------')
display(df.groupby(df['division'])['officer_age'].mean())
df.groupby(df['division'])['officer_age'].mean().plot.barh(figsize=(7.5,12))
plt.title('Avg Age of Officer by Division')
plt.show()

print('--------------------------------------------')
print('Officer Injury Rates by their Race/Ethnicity')
print('--------------------------------------------')
off_ethn_injd = pd.crosstab(df['officer_race/ethnicity'], df['officer_injured'])
display(off_ethn_injd)

off_ethn_injd.plot.bar(stacked=True, 
                       rot=60)
plt.title('Race/Ethnicity of Officers and their Injury Rates')
plt.show()

off_age_ethn = df.groupby(df['officer_race/ethnicity'])['officer_age'].mean()

display(off_age_ethn)

off_age_ethn.plot.bar(rot=60)
plt.title('Avg Officer Age by Race/Ethnicity')
plt.show()

off_age_ethn = df.groupby(df['officer_race/ethnicity'])['officer_age'].count()

print('---------------------------------------------')
print('Total Officers on the Force by Race/Ethnicity')
print('---------------------------------------------')
display(off_age_ethn)

off_age_ethn.plot.bar(rot=60)
plt.title('Total Officers by Race/Ethnicity')
plt.show()


# ### Examining the subject data according to a few variables...

# In[8]:


display(df.groupby(df['subject_arrested'])['subject_age'].mean())
df.groupby(df['subject_arrested'])['subject_age'].mean().plot.bar(rot=60)
plt.title('Averae Age of Suspect by Arrest Status')
plt.show()


wthr_arrst = pd.crosstab(df.weather_condition, df.subject_arrested)
display(wthr_arrst)

pd.crosstab(df.weather_condition, df.subject_arrested).plot.bar(stacked=True, 
                                                                rot=60)
plt.show()

# svc_ethn = pd.crosstab(df.service_type, df.subject_ethnicity)
#display(svc_ethn)

#svc_ethn.plot.bar(stacked=True, 
#                  rot=60)
#plt.title('Suspect Ethnicity by Service Type')
#plt.ylabel('Total Incidents')
#plt.show()

sbj_arrst_ethn = pd.crosstab(df.subject_ethnicity, df.subject_arrested)
display(sbj_arrst_ethn)

sbj_arrst_ethn.plot.bar(stacked=True, 
                         rot=60)
plt.title('Suspect Arrest Rates by Race')
plt.xlabel('Suspect Race')
plt.show()

sbjct_hosp_race = pd.crosstab(df.subject_ethnicity, df.subject_hospitalized)
display(sbjct_hosp_race)

sbjct_hosp_race.plot.bar(stacked=True, 
                         rot=60)
plt.title('Suspect Hospitalization Rates by Race')
plt.xlabel('Suspect Race')
plt.show()


# ### Comparing officer and subject numerical data...

# In[9]:


sns.jointplot('officer_age', 
              'subject_age', 
              data=df, 
              kind='hex')
plt.title('Officer and Suspect Age')
plt.show()

sns.jointplot('officer_years_of_service', 
              'officer_age', 
              data=df, 
              kind='hex')
plt.title('Officer Age and Years of Service')
plt.show()

sns.jointplot('officer_years_of_service', 
              'subject_age', 
              data=df, 
              kind='hex')
plt.title('Officer Years of Service and Subject Age')
plt.show()


# Hmmmm....seems like NOLA PD's only inclined to make arrests when the weather's good...I wander why that is? Maybe subjects are less likely to resist arrest in bad weather?  
