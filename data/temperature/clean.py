import sys 
sys.path.append('../../')

import usStates
import numpy as np
import pandas as pd


# Initialize a dataframe
df_0 = pd.DataFrame(np.arange(1999,2014), columns = ['year'])

# Raw data for each state are saved one file per state, so we need to combine them
for i in range(1,51):
    fName = '%02d.csv' % i

    with open(fName, 'r') as FH:
        state = FH.readline().split(',')[0]
    
    tmp_df = pd.read_csv(fName, skiprows = 4, usecols = [1], header = None)
    
    df_0[state] = tmp_df.values

# convert sates to two letter code
df_0.columns = [usStates.nameToCode_dict[i] if i in usStates.nameToCode_dict else i for i in df_0.columns.values]
df_0.rename(columns={'D.C.': 'DC'}, inplace=True)


df_0.set_index('year', inplace=True)

'''
Initially, the data has 'states' as rows and 'years' as columns. 
I converted it to have three columns, 'state', 'year', 'temperature'.
Then, I can extract the 'temperature' column and join with other data based on 'state' and 'year'
'''

df_1 = df_0.stack()
df_2 = df_1.reset_index()
df_2.columns = ['year', 'state', 'temperature']

df_3 = df_2.sort_values(['state', 'year'])
df_3 = df_3.loc[:,['state', 'year', 'temperature']]
df_3['year'] = df_3['year'].astype(int)
df_3['state'] = df_3['state'].astype(str)
df_3['temperature'] = df_3['temperature'].astype(float)

# Export data to a csv file
df_3.to_csv('clean.csv', index=False)


