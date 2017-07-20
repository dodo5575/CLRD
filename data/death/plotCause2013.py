import sys
sys.path.append('../../')

import usStates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns



df_0 = pd.read_csv('NCHS_-_Age-adjusted_Death_Rates_for_the_Top_10_Leading_Causes_of_Death__United_States__2013.csv', 
usecols = [0,2,3,4]) 

df_1 = df_0.loc[ (df_0.YEAR == 2013) & (df_0.CAUSE_NAME != 'All Causes'), :].copy()

df_1['DEATHS'] = df_1['DEATHS'].astype(int)

df_2 = df_1.loc[:,['CAUSE_NAME', 'DEATHS']].groupby('CAUSE_NAME').sum()
df_2 = df_2.sort_values('DEATHS')


fig, ax = plt.subplots()

df_2.plot.barh()

plt.title('Top 10 leading causes of death in 2013 U.S.')
plt.xlabel('Death')
plt.ylabel('Cause')


plt.savefig('cause2013.pdf', transparent=True, bbox_inches='tight')
plt.savefig('cause2013.png', transparent=True, bbox_inches='tight')
#plt.show()
fig.clf()
plt.clf()
plt.close('all')


