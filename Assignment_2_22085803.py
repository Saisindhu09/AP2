# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 16:57:52 2023

@author: saisi
"""

# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#setting views for dataframe
pd.options.display.max_rows = 40
pd.options.display.max_columns = 40


def loading_dataframe(filename):
    """
    Function to load the dataframe and manipulate the country and rows
    features and return two datframes.
    """

    df_year_col = pd.read_excel(filename, engine="openpyxl")
    df_temp = pd.melt(
        df_year_col,
        id_vars=[
            'Country Name',
            'Country Code',
            'Indicator Name',
            'Indicator Code'],
        var_name='Year',
        value_name='Value')
    df_country_col = df_temp.pivot_table(
        index=['Year', 'Country Code', 'Indicator Name', 'Indicator Code'],
        columns='Country Name',
        values='Value').reset_index()
    df_country_col = df_country_col.drop_duplicates().reset_index()
    return df_year_col, df_country_col


def years_to_analyze_data1(df, start_year, end_year, frequency=5):
    """
    function retrieve data from start year to end year based on interval.
    """

    df_temp = df.copy()
    years_to_analyze=[i for i in range(start_year, end_year, frequency)]
    col_to_keep=['Country Name','Indicator Name']
    col_to_keep.extend(years_to_analyze)
    df_temp =  df_temp[col_to_keep]
    df_temp = df_temp.dropna(axis=0, how="any")
    return df_temp


def data_specific_feature(df, indicator, values_list):
    """
    function to filter dataframe based on values of column.
    """
    df_temp = df.copy()
    df_field = \
        df_temp[df_temp[indicator].isin(values_list)].reset_index(drop=True)
    return df_field


def bar_plot_countrywise_year(df, indicator):
    """
    function to plot the bar chart of a feature for some countries yearwise
    """

    # Filter numeric columns for plotting
    df = df.copy()
    df.set_index('Country Name', inplace=True)
    numeric_columns = df.columns[df.dtypes == 'float64']
    df_numeric = df[numeric_columns]

    # Plotting
    plt.figure(figsize=(50, 50))
    df_numeric.plot(kind='bar')
    plt.title(indicator)
    plt.xlabel('Country Name')
    plt.ylabel(indicator)
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='lower left')
    plt.show()


def get_data_by_indicator(data1):
    """
    function to give indicator names as columns for respective
    dataframe country
    """
    df=data1.copy()
    df_melted = df.melt(
        id_vars='Indicator Name',
        var_name='Year',
        value_name='Value')
    df_pivoted = df_melted.pivot(
        index='Year',
        columns='Indicator Name',
        values='Value')
    df_pivoted.reset_index(inplace=True)
    df_pivoted = df_pivoted.apply(pd.to_numeric, errors='coerce')
    del df_pivoted['Year']
    df_pivoted = df_pivoted.rename_axis(None, axis=1)
    return df_pivoted


def time_series_plot(data, indicator):
    """
    plot year wise data as line for respective countries
    """
    df = data.copy()
    # Filter numeric columns for plotting
    df.set_index('Country Name', inplace=True)
    numeric_columns = df.columns[df.dtypes == 'float64']
    df_numeric = df[numeric_columns]

    # Plotting
    plt.figure(figsize=(12, 6))
    for country in df_numeric.index:
        plt.plot(
            df_numeric.columns,
            df_numeric.loc[country],
            label=country, linestyle='dashed', marker='s')

    plt.title(indicator)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='lower left')
    plt.show()


# Reading the data
df_year_col,df_country_col = loading_dataframe('world_bank_climate.xlsx')

# Taking data from 1990 to 2020 on 4 year gap.
df_year_col_temp = years_to_analyze_data1(df_year_col, 1990, 2021, 6)
countries = \
    df_year_col_temp['Country Name'].value_counts().index.tolist()[8:15]
# Use of describe method
print(df_year_col_temp.describe())

df_year_col_mortality = data_specific_feature(
    df_year_col_temp,
    'Indicator Name',
    ['Mortality rate, under-5 (per 1,000 live births)'])
# Use of describe method
print(df_year_col_mortality.describe())

df_year_col_mortality  = data_specific_feature(
    df_year_col_mortality,'Country Name', countries)
# Use of describe method
print(df_year_col_mortality.describe())
# Bar plot
bar_plot_countrywise_year(
    df_year_col_mortality, 'Mortality rate, under-5 (per 1,000 live births)')


df_year_col_arable_land = data_specific_feature(
    df_year_col_temp,
    'Indicator Name',
    ['Arable land (% of land area)'])
df_year_col_arable_land  = data_specific_feature(
    df_year_col_arable_land, 'Country Name', countries)
print(df_year_col_arable_land.describe())
bar_plot_countrywise_year(
    df_year_col_arable_land,
    '"Arable land (% of land area)"')

# East Asia & PAcific Region Data
df_year_col_east_asia_pacific = data_specific_feature(
    df_year_col_temp,
    'Country Name',
    ['East Asia & Pacific'])
data_heat_map_east_asia_pacific = get_data_by_indicator(df_year_col_east_asia_pacific)
features_check = [
    "CO2 emissions (metric tons per capita)",
    "Urban population (% of total population)",
    "Total greenhouse gas emissions (kt of CO2 equivalent)",
    "Agricultural land (% of land area)",
    "Arable land (% of land area)",
    "Renewable energy consumption (% of total final energy consumption)"]
data_heat_map_east_asia_pacific_sub = data_heat_map_east_asia_pacific[features_check]
print(data_heat_map_east_asia_pacific_sub.corr())
sns.heatmap(
    data_heat_map_east_asia_pacific_sub.corr(),
    annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')

df_year_col_ghg= data_specific_feature(
    df_year_col_temp,
    'Indicator Name',
    ['Total greenhouse gas emissions (kt of CO2 equivalent)'])
df_year_col_ghg  = data_specific_feature(
    df_year_col_ghg, 'Country Name', countries)
print(df_year_col_ghg.describe())

df_year_col_ub_pop= data_specific_feature(
    df_year_col_temp,
    'Indicator Name',
    ['Urban population (% of total population)'])
df_year_col_ub_pop  = data_specific_feature(
    df_year_col_ub_pop, 'Country Name', countries)
time_series_plot(
    df_year_col_ghg,
    'Total greenhouse gas emissions (kt of CO2 equivalent)')

df_year_col_ag_land= data_specific_feature(
    df_year_col_temp,
    'Indicator Name',
    ['Agricultural land (% of land area)'])
df_year_col_ag_land  = data_specific_feature(
    df_year_col_ag_land,
    'Country Name',
    countries)
print(df_year_col_ag_land.describe())
time_series_plot(df_year_col_ag_land,'Agricultural land (% of land area)')

# Data of Germany country
df_year_col_germany = data_specific_feature(
    df_year_col_temp, 'Country Name', ['Germany'])
data_heat_map_germany = get_data_by_indicator(df_year_col_germany)
data_heat_map_germany_sub = data_heat_map_germany[[
    "CO2 emissions (metric tons per capita)",
    "Urban population (% of total population)",
    "Total greenhouse gas emissions (kt of CO2 equivalent)",
    "Agricultural land (% of land area)",
    "Arable land (% of land area)",
    "Renewable energy consumption (% of total final energy consumption)"]]
sns.heatmap(
    data_heat_map_germany_sub.corr(),
    annot=True, cmap='inferno', linewidths=.5, fmt='.3g')

# Data of USA
df_year_col_us = data_specific_feature(
    df_year_col_temp, 'Country Name', ['United States'])
data_heat_map_us = get_data_by_indicator(df_year_col_us)
data_heat_map_us_sub = data_heat_map_us[[
    "CO2 emissions (metric tons per capita)",
    "Urban population (% of total population)",
    "Total greenhouse gas emissions (kt of CO2 equivalent)",
    "Agricultural land (% of land area)",
    "Arable land (% of land area)",
    "Renewable energy consumption (% of total final energy consumption)"]]
plt.figure()
sns.heatmap(
    data_heat_map_us_sub.corr(),
    annot=True, cmap='inferno', linewidths=.5, fmt='.3g')
