import sys
sys.path.append('../../')

import usStates
import numpy as np
import pandas as pd


# Read raw data
df_0 = pd.read_csv('NCHS_-_Age-adjusted_Death_Rates_for_the_Top_10_Leading_Causes_of_Death__United_States__2013.csv', 
usecols = [0,2,3,4]) 

# Extract only death from CLRD
df_0 = df_0.loc[ df_0.CAUSE_NAME == 'CLRD', ['STATE','YEAR', 'DEATHS']]

# Rename columns
df_0.columns = ['state','year', 'death'] 

# Convert sates to two letter code
df_0.loc[:, 'state'] = [usStates.nameToCode_dict[i] if i in usStates.nameToCode_dict else i for i in df_0.state.values]

# Remove data from the sum of the U.S.
df_1 = df_0.loc[df_0.state != 'United States', :].copy()

# Sort by state and year
df_2 = df_1.sort_values(['state', 'year'])

#print( df_2 )

# Export data to a csv file
df_2.to_csv('clean.csv', index=False)


