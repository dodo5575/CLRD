import numpy as np
import pandas as pd


# Read the raw data into a dataframe
df_0 = pd.read_csv('population.csv') 

# Process the column names
df_0.rename(columns={'Unnamed: 0': 'year'}, inplace=True)

# Take only data in 1999~2013
df_0 = df_0[ (df_0.year > 1998) & (df_0.year < 2014) ] 


df_0.set_index('year', inplace=True)

'''
Initially, the data has 'states' as rows and 'years' as columns. 
I converted it to have three columns, 'state', 'year', 'population'.
Then, I can extract the 'population' column and join with other data based on 'state' and 'year'
'''
df_1 = df_0.stack()
df_2 = df_1.reset_index()
df_2.columns = ['year', 'state', 'population']

df_3 = df_2.sort_values(['state', 'year'])
df_3 = df_3.loc[:,['state', 'year', 'population']]
df_3['year'] = df_3['year'].astype(int)
df_3['state'] = df_3['state'].astype(str)
df_3['population'] = df_3['population'].astype(int)

# Export data to a csv file
df_3.to_csv('clean.csv', index=False)


