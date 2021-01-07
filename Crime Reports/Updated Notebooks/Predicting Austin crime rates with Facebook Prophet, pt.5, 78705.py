#!/usr/bin/env python
# coding: utf-8

# # Predicting crime rates with Facebook Prophet 
# 
# ## Modeling crime in zip code 78705

# In[1]:


# importing necessary libraries and configurations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot

plt.style.use("seaborn")
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# loading data
df = pd.read_csv("df_05.csv")


# In[3]:


# data cleaning and indexing
drop = ["zip_code", "latitude", "longitude"]

df.drop(drop, axis=1, inplace=True)
df.occurred_date = df.occurred_date.astype("datetime64")
df.set_index(["occurred_date"], inplace=True)
df.sort_index(inplace=True)


# ## Modeling the data as-is

# In[5]:


# Prepping to forecast
df_fbprophet = df.copy()

df_m = df_fbprophet.resample("D").size().reset_index()
df_m.columns = ["date", "daily_crime_count"]
df_m_final = df_m.rename(columns={"date": "ds", "daily_crime_count": "y"})
df_m_final["y"] = pd.to_numeric(df_m_final["y"])
y = df_m_final["y"].to_frame()
y.index = df_m_final["ds"]
n = np.int(y.count())

m = Prophet(interval_width=0.95)
m.add_seasonality(name="monthly", period=30.5, fourier_order=10)
m.add_seasonality(name="weekly", period=52, fourier_order=10)
m.add_seasonality(name="daily", period=365, fourier_order=10)
m.add_country_holidays(country_name="US")
m.fit(df_m_final)

future = m.make_future_dataframe(periods=365, freq="D")

forecast = m.predict(future)

fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

fig2 = m.plot_components(forecast)

fig2_1 = plot_plotly(m, forecast)

display(fig)
display(fig2)
display(fig2_1)


# In[6]:


# The forecast is 'log transformed', so we need to 'inverse' it back by using the exp
forecast_df_exp = np.exp(forecast[["yhat", "yhat_lower", "yhat_upper"]])
forecast_df_exp.index = forecast["ds"]

# Calculating MAPE error 
error = forecast_df_exp["yhat"] - y["y"]
MAPE_df = (error / y["y"]).abs().sum() / n * 100
round(MAPE_df, 2)


# ## Dealing with outliers and their effects on model accuracy

# In[8]:


# Make another copy of the data frame as m2
df_prophet2 = df_m_final.copy()

# Define the Upper Control Limit and Lower Control Limit as 3 standard deviations from the mean
ucl = df_prophet2.mean() + df_prophet2.std() * 3
lcl = df_prophet2.mean() - df_prophet2.std() * 3

# display the number of outliers found
print(
    "Above 3 standard deviations: ",
    df_prophet2[df_prophet2["y"] > ucl["y"]]["y"].count(),
    "entries",
)
print(
    "Below 3 standard deviations: ",
    df_prophet2[df_prophet2["y"] < lcl["y"]]["y"].count(),
    "entries",
)

# Remove them by setting their value to None. Prophet says it can handle null values.
df_prophet2.loc[df_prophet2["y"] > ucl["y"], "y"] = None
df_prophet2.loc[df_prophet2["y"] < lcl["y"], "y"] = None

# Log transformation
df_prophet2["y"] = pd.to_numeric(df_prophet2["y"])
# Run Prophet using model 2
m2_no_outlier = Prophet(interval_width=0.95)
m2_no_outlier.add_seasonality(name="monthly", period=30.5, fourier_order=10)
m2_no_outlier.add_seasonality(name="weekly", period=52, fourier_order=10)
m2_no_outlier.add_seasonality(name="daily", period=365, fourier_order=10)
m2_no_outlier.add_country_holidays(country_name="US")
m2_no_outlier.fit(df_prophet2)
future = m2_no_outlier.make_future_dataframe(periods=365)
forecast_m2 = m2_no_outlier.predict(future)
fig_m2 = m2_no_outlier.plot(forecast_m2)
a = add_changepoints_to_plot(fig_m2.gca(), m2_no_outlier, forecast_m2)

fig2m2 = m2_no_outlier.plot_components(forecast_m2)

fig2_1m2 = plot_plotly(m2_no_outlier, forecast_m2)

display(fig_m2)
display(fig2m2)
display(fig2_1m2)


# In[9]:


# The forecast is 'log transformed', so we need to 'inverse' it back by using the exp
forecast_m2_exp = np.exp(forecast_m2[["yhat", "yhat_lower", "yhat_upper"]])
forecast_m2_exp.index = forecast_m2["ds"]

# Calculate the error
error = forecast_m2_exp["yhat"] - y["y"]
MAPE_m2 = (error / y["y"]).abs().sum() / n * 100
round(MAPE_m2, 2)


# In[ ]:




