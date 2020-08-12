#!/usr/bin/env python
# coding: utf-8

# # Austin Crime Reports Analysis, Pt.2
# 
# After completing the first notebook, I did some further analysis in Excel with parts of the dataset I saved to individual csv files. In particular, I separated the address column into street numbers and names to add some dimension.

# In[1]:


# Importing essential libraries and configurations
get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


# In[2]:


# Loading the dataframes
df_mur = pd.read_csv('df_mur.csv')
df_mur_cap = pd.read_csv('df_mur_cap.csv')
df_rape = pd.read_csv('df_rape.csv')


# In[3]:


df_mur.report_date_time = df_mur.report_date_time.astype('datetime64')
df_mur.set_index(['report_date_time'], inplace=True)
df_mur.sort_index(inplace=True)

df_mur_cap.report_date_time = df_mur_cap.report_date_time.astype('datetime64')
df_mur_cap.set_index(['report_date_time'], inplace=True)
df_mur_cap.sort_index(inplace=True)

df_rape.report_date_time = df_rape.report_date_time.astype('datetime64')
df_rape.set_index(['report_date_time'], inplace=True)
df_rape.sort_index(inplace=True)

# Converting columns to their appropriate data types
df_mur.clearance_date = df_mur.clearance_date.astype('datetime64')
df_mur.occurred_date = df_mur.occurred_date.astype('datetime64')
df_mur.highest_offense_code = df_mur.highest_offense_code.astype('int64')

df_mur_cap.clearance_date = df_mur_cap.clearance_date.astype('datetime64')
df_mur_cap.occurred_date = df_mur_cap.occurred_date.astype('datetime64')
df_mur_cap.highest_offense_code = df_mur_cap.highest_offense_code.astype('int64')

df_rape.clearance_date = df_rape.clearance_date.astype('datetime64')
df_rape.occurred_date = df_rape.occurred_date.astype('datetime64')
df_rape.highest_offense_code = df_rape.highest_offense_code.astype('int64')


# ## How are violent crimes distributed...

# ### Question 1. Are there any addresses in Austin, known to be hotspots for violent crime? 
# 
# Note: For non-capital murder, in the following code, I am only including results that have hosted 2 incidents or more. For rape, I am only including the top 50 results.

# In[4]:


print('-----------------------------------------------')
print('Addresses where violent crimes most often occur')
print('-----------------------------------------------')
print('Murder')
print('----------------------------------')
display(df_mur.address.value_counts().head(22))
print('----------------------------------')
display(df_mur.address.value_counts(normalize=True).head(22))

print('----------------------------------')
print('Capital Murder')
print('----------------------------------')
display(df_mur_cap.address.value_counts())
print('----------------------------------')
display(df_mur_cap.address.value_counts(normalize=True))
print('----------------------------------')
print('Rape')
print('----------------------------------')
display(df_rape.address.value_counts().head(50))
print('----------------------------------')
display(df_rape.address.value_counts(normalize=True).head(50))


# ### Question 2. Are there any specific streets in Austin, that are hotspots for violent crime? 
# 
# Note: For murder only, in the following code, I am only including results that >= 1%...

# In[5]:


print('---------------------------------------------')
print('Streets where violent crimes most offen occur')
print('---------------------------------------------')

print('----------------------------------')
print('Murder')
print('----------------------------------')
display(df_mur.street_name.value_counts().head(10))
print('----------------------------------')
display(df_mur.street_name.value_counts(normalize=True).head(10))
print('----------------------------------')
print('Capital Murder')
print('----------------------------------')
display(df_mur_cap.street_name.value_counts())
print('----------------------------------')
display(df_mur_cap.street_name.value_counts(normalize=True))
print('----------------------------------')
print('Rape')
print('----------------------------------')
display(df_rape.street_name.value_counts())
print('----------------------------------')
display(df_rape.street_name.value_counts(normalize=True))


# ### Question 3. In what location type is violent crime most likely to occur? 
# Note: For non-capital murder, and rape, I am only including results that >= 1% of the total.

# In[6]:


print('----------------------------------------------------')
print('Location types where violent crimes most often occur')
print('----------------------------------------------------')
print('Murder')
print('----------------------------------')
display(df_mur.location_type.value_counts().head(9))
print('----------------------------------')
display(df_mur.location_type.value_counts(normalize=True).head(9))
print('----------------------------------')
print('Capital Murder')
print('----------------------------------')
display(df_mur_cap.location_type.value_counts())
print('----------------------------------')
display(df_mur_cap.location_type.value_counts(normalize=True))
print('----------------------------------')
print('Rape')
print('----------------------------------')
display(df_rape.location_type.value_counts().head(8))
print('----------------------------------')
display(df_rape.location_type.value_counts(normalize=True).head(8))


# **Summary of Violent Crimes**
# 
# For NON-CAPITAL MURDER, analysis indicates 3 separate addresses which played host to at least 3 counts each, since 2003: 
# 
# 1.	4700 E Riverside Dr
# 2.	8610 N Lamar Blvd 
# 3.	8800 N IH 35, Svrd SB. 
# 
# A Google search of the addresses indicated the first address home to the Tempo Apartment Complex. The second is the location of what appears to be a strip mall. The third is home to the Starburst Apartment Complex. 
# 
# In addition, 2 separate addresses played host to at least 2 counts of CAPITAL MURDER each: 
# 
# 1.	815 W Slaughter Lane
# 2.	7000 Decker Lane 
# 
# ***Note: Remember that CAPITAL murder means that the defendant(s), based on his/her/their motivations and actions during his/her/their crime commission, makes him/her/them automatically eligible for the death sentence, under Texas law.***
# 
# Interestingly, rape occurred, or was reported as having occurred, an astonishing 51 separate times at the 900 Block of E 32nd Street, since 2003.
# 
# ***Note: Seton Hospital is located here. Why is this? Possibly, because rape often coincides with other forms of violence against the victim(s) and so the victim(s) were reporting from a hospital bed?***  
# 
# At any rate, further research is necessary to determine what the reason(s) are with more assurance. 
# 
# Further, when we break down violent crimes by the street on which they occurred, we see that most non-capital murders occurred on E Riverside Dr, a total of 13 counts. However, 4 separate capital murders occurred on N Lamar Blvd, as well as 10 separate non-capital murders, which makes N Lamar Blvd the number 1 hotspot for murder in general.   
# 
# In terms of individual streets, rape occurred 95 separate times somewhere along N IH 35 SVRD SB, specifically. N Lamar Blvd comes in at number 2, with 80 separate counts of rape at one point or another along that route. 
# 
# Another street of note is Wickersham Ln which has hosted 53 separate counts of rape and 4 separate counts of non-capital murder, since 2003. 
# 
# Finally, we see that home is not necessarily where one is the safest…
# 
# 42.56% of non-capital murder, nearly 60% of capital murder, and over 61% of rape all occur within a residence or home. 
# 
# As a member of the LGBT Community, I particularly remember an incident from 2013, in which a man, David Villareal, brought home another man he met at Rain Nightclub on 4th Street. The man he brought home, Matthew Bacon, proceeded to bludgeon and eviscerate him to death, before stealing his belongings and making his getaway in David’s vehicle. Matthew has since received 60 years for his crimes, for which he avoided the death sentence by confessing. 
# 
# The lesson, ladies and gentlemen: BE CAREFUL WHAT STRANGER(S) YOU INVITE INTO YOUR HOME!!
# 
# 
#  
# 

# In[8]:


frames = [df_mur, df_mur_cap, df_rape]

df = pd.concat(frames)
df.sort_index(inplace=True)


# In[16]:


df.address.value_counts()

