import sys
sys.path.append('../../')

import usStates
import numpy as np
import pandas as pd


# Create column names for the dataframe
columns_list = np.arange(2013,1998, -1).tolist()
columns_list.insert(0,'state')

# Read input data to a dataframe using the created column names
df_0 = pd.read_csv('MedianHouseholdIncome.csv', engine='python', 
skiprows=6, skipfooter=57, 
usecols = np.append([0], np.delete(np.arange(5,36,2), 1)), 
header=None, names = columns_list)

# convert sates to two letter code
df_0.loc[:, 'state'] = [usStates.nameToCode_dict[i] if i in usStates.nameToCode_dict else i for i in df_0.state.values]
df_0.loc[9,'state'] = 'DC'


df_0.set_index('state', inplace=True)

'''
Initially, the data has 'states' as rows and 'years' as columns. 
I converted it to have three columns, 'state', 'year', 'income'.
Then, I can extract the 'income' column and join with other data based on 'state' and 'year'
'''
df_1 = df_0.stack()
df_2 = df_1.reset_index()
df_2.columns = ['state', 'year', 'income']

df_3 = df_2.loc[df_2.state != 'United States', :].copy()
df_3 = df_3.sort_values(['state', 'year'])
df_3['year'] = df_3['year'].astype(int)
df_3['state'] = df_3['state'].astype(str)
df_3['income'] = df_3['income'].astype(int)


# Export data to a csv file
df_3.to_csv('clean.csv', index=False)


