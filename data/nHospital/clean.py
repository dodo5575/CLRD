import sys 
sys.path.append('../../')

import usStates
import numpy as np
import pandas as pd


# Read data into a dataframe
df_0 = pd.read_csv('nHospital_1999_2015.csv', engine='python', 
skiprows=2, skipfooter=10, 
usecols = np.arange(0,16)) 

# Process the column names
df_0.columns = [col[:4] for col in df_0.columns.values]
df_0.rename(index = str, columns={'Loca': 'state'}, inplace=True)

# convert sates to two letter code
df_0.loc[:, 'state'] = [usStates.nameToCode_dict[i] if i in usStates.nameToCode_dict else i for i in df_0.state.values]


df_0.set_index('state', inplace=True)


'''
Initially, the data has 'states' as rows and 'years' as columns. 
I converted it to have three columns, 'state', 'year', 'nHospital'.
Then, I can extract the 'nHospital' column and join with other data based on 'state' and 'year'
'''
df_1 = df_0.stack()
df_2 = df_1.reset_index()
df_2.columns = ['state', 'year', 'nHospital']

df_3 = df_2.loc[df_2.state != 'United States', :].copy()
df_3 = df_3.sort_values(['state', 'year'])
df_3['year'] = df_3['year'].astype(int)
df_3['state'] = df_3['state'].astype(str)
df_3['nHospital'] = df_3['nHospital'].astype(int)


# Export data to a csv file
df_3.to_csv('clean.csv', index=False)


