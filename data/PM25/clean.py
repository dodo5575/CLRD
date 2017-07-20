import sys 
sys.path.append('../../')

import usStates
import numpy as np
import pandas as pd


# Extract and combine the PM2.5 data
for i in range(1999,2014):

    # Load the csv file and select only necessary columns
    tmp_df_0 =  pd.read_csv('annual_conc_by_monitor_%d.csv' % i, usecols = [8,10,16,50])

    # Filter unwanted data entry    
    tmp_df_1 = tmp_df_0[ (tmp_df_0['Parameter Name'] == 'PM2.5 - Local Conditions') \
    & (tmp_df_0['Pollutant Standard'] == 'PM25 Annual 2012') \
    & ((tmp_df_0['State Name'] != 'Puerto Rico') & (tmp_df_0['State Name'] != 'Virgin Islands'))]
    
    # Select only the name of the state and the observed PM2.5 data
    tmp_df_2 = tmp_df_1[['State Name', 'Observation Count']].copy()
    tmp_df_2.columns = ['state', i]

    # Use df_0 to collect data from all years. 
    # If this is the first year, initialize df_0. Else, append the data to df_0    
    if 'df_0' not in locals():
        df_0 = tmp_df_2.groupby('state').mean()
    else:
        df_0[i] = tmp_df_2.groupby('state').mean()[i]

# convert sates to two letter code
df_0.index = [usStates.nameToCode_dict[i] if i in usStates.nameToCode_dict else i for i in df_0.index.values]
df_0.rename(index={'District Of Columbia': 'DC'}, inplace=True)


'''
Initially, the data has 'states' as rows and 'years' as columns. 
I converted it to have three columns, 'state', 'year', 'PM25'.
Then, I can extract the 'PM25' column and join with other data based on 'state' and 'year'
'''
df_1 = df_0.stack()
df_2 = df_1.reset_index()
df_2.columns = ['state', 'year', 'PM25']

df_3 = df_2.sort_values(['state', 'year'])
df_3['year'] = df_3['year'].astype(int)
df_3['state'] = df_3['state'].astype(str)
df_3['PM25'] = df_3['PM25'].astype(float)


# Export data to a csv file
df_3.to_csv('clean.csv', index=False)


