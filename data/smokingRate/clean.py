import sys
sys.path.append('../../')

import usStates
import numpy as np
import pandas as pd

## Read the first dataset
# Load the raw data to a dataframe
tus_df_0 = pd.read_csv('smokingRate_TUS-CPS.csv')

# Trim the 'Year' column
tus_df_1 = tus_df_0.copy()
tus_df_1.loc[:,'Year'] = [i.strip() for i in tus_df_1.Year.values]

# Some of the years are averaged, ex: 1998-1999
# The following code find each of them and add rows for each using the averaged data, ex: add rows with year 1998 and 1999
years = np.unique(tus_df_1.Year.values)
years_multi = [i for i in years if len(i) > 4]

for j in years_multi:
    for i in j.split('-'):
        tmp_df = tus_df_1.loc[tus_df_1.Year == j, :].copy()
        tmp_df.loc[:,'Year'] = i 
        tus_df_1 = tus_df_1.append(tmp_df, ignore_index=True)

tus_df_2 = tus_df_1.loc[tus_df_1['Year'].str.len() == 4, :]
tus_df_2 = tus_df_2.sort_values(['Location Description', 'Year'])


# Extract the data with current smoker despite their gender
tus_df_3 = tus_df_2.loc[(tus_df_2.Gender == 'Overall') & (tus_df_2.Response == 'Current'), ['Year', 'Location Description', 'Data Value']]

# Rename the columns
tus_df_3.rename(index=str, columns={'Location Description': 'state', 'Data Value': 'smokingRate', 'Year': 'year'}, inplace=True)



## Read the second dataset
brfss_df_0 = pd.read_csv('smokingRate_BRFSS.csv')

# Extract the data with current smoker despite their gender
brfss_df_1 = brfss_df_0.loc[(brfss_df_0.Gender == 'Overall') & (brfss_df_0.Response == 'Current') & ((brfss_df_0.Year == 2012) | (brfss_df_0.Year == 2013)), ['Year', 'Location Description', 'Data Value']]
# Rename the columns
brfss_df_1.rename(index=str, columns={'Location Description': 'state', 'Data Value': 'smokingRate', 'Year': 'year'}, inplace=True)


# Combine the two data frame into one dataframe
smoking_df = tus_df_3.append(brfss_df_1, ignore_index=True)

# convert sates to two letter code
smoking_df.loc[:,'state'] = [usStates.nameToCode_dict[i] if i in usStates.nameToCode_dict else i for i in smoking_df.state.values]


smoking_df['year'] = smoking_df['year'].astype(int)
smoking_df['state'] = smoking_df['state'].astype(str)
smoking_df['smokingRate'] = smoking_df['smokingRate'].astype(float)

# Sort the dataframe by state and year
smoking_df = smoking_df.sort_values(['state', 'year'])

# Extract data within 1999~2013 and not in Puerto Rico or Guam or National Median.
smoking_df = smoking_df[ (smoking_df.year < 2014) & (smoking_df.year > 1998)] 
smoking_df = smoking_df[ (smoking_df.state != 'Puerto Rico') & (smoking_df.state != 'National Median (States and DC)') & (smoking_df.state != 'Guam')] 

smoking_df = smoking_df.loc[:,['state', 'year', 'smokingRate']]

# Export data to a csv file
smoking_df.to_csv('clean.csv', index=False)



