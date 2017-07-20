import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, SGDRegressor


# First, load the target data into a dataframe
data_df = pd.read_csv('data/death/clean.csv')


# For each features, load the associated CSV file and join to the target dataframe 
for i in ['income', 'nHospital', 'PM25', 'population', 'smokingRate', 'temperature']:
    tmp_df = pd.read_csv('data/%s/clean.csv' % i)

    data_df = pd.merge(data_df, tmp_df, how='left', on=['state', 'year'])


# Remove rows that have missing data
data_complete_df = data_df.dropna().copy() 


# Create new features hostipal density "hospitalD" and "deathRate"
data_complete_df['hospitalD'] = data_complete_df.nHospital / data_complete_df.population
data_complete_df['deathRate'] = data_complete_df.death / data_complete_df.population


# Drop unnecessary columns for later processing
data_trunc_df = data_complete_df.drop([ 'death', 'nHospital', 'population'], axis=1)


#-------------------------------
# Plotting scatter matrix
#-------------------------------

fig, ax = plt.subplots()
sm = scatter_matrix(data_trunc_df.iloc[:,2:], alpha=0.2, figsize=(7, 7), diagonal='kde')

# Change label rotation
[s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]

# May need to offset label when rotating to prevent overlap of figure
[s.get_yaxis().set_label_coords(-0.55,0.5) for s in sm.reshape(-1)]

# Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]

plt.savefig('scatter_matrix.pdf', transparent=True, bbox_inches='tight')
plt.savefig('scatter_matrix.png', transparent=True, bbox_inches='tight')
#plt.show()
fig.clf()
plt.clf()
plt.close('all')


#-------------------------------
# Regression
#-------------------------------

# Extract the data to an array 
data_Array = data_trunc_df.iloc[:,2:].values


# Standardization: scale data to have 0 mean and unit std 
scalerX = preprocessing.StandardScaler().fit(data_Array[:,:-1])
X_scaled = scalerX.transform(data_Array[:,:-1])

scalerY = preprocessing.StandardScaler().fit(data_Array[:,-1])
Y_scaled = scalerY.transform(data_Array[:,-1])


# Randomly split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, Y_scaled,
test_size=0.33, random_state=42)


## Linear regression
# Create linear regression object
linreg = LinearRegression()

# Train the model using the training sets
linreg.fit(X_train, y_train)
print('Linear regression\nscore: ', linreg.score(X_train, y_train))


#-------------------------------
# Plotting feature importance
#-------------------------------

fig, ax = plt.subplots()
plt.bar(np.arange(5), np.absolute(linreg.coef_[np.argsort(np.absolute(linreg.coef_))]), 0.5, color = plt.cm.Blues(np.linspace(0.3, 1, 5)))
plt.xticks(np.arange(5), data_trunc_df.iloc[:,2:-1].columns.values[np.argsort(np.absolute(linreg.coef_))], rotation=30)
plt.ylabel('Coefficient absolute value')

plt.savefig('features.pdf', transparent=True, bbox_inches='tight')
plt.savefig('features.png', transparent=True, bbox_inches='tight')
#plt.show()
fig.clf()
plt.clf()
plt.close('all')


# Calculate RMSE
err = linreg.predict(X_train) - y_train
rmse_train = np.sqrt(np.dot(err,err) / len(y_train))

# Now let's compute RMSE using 10-fold x-validation
kf = KFold(n_splits=10)
xval_err = 0
for train, test in kf.split(X_train):
    #print(train, test)
    linreg.fit(X_train[train], y_train[train])
    p = linreg.predict(X_train[test])
    e = p - y_train[test]
    xval_err += np.dot(e,e)

rmse_10cv = np.sqrt(xval_err / len(X_train))
print('RMSE on training: %.4f' % rmse_train)
print('RMSE on 10-fold CV: %.4f' % rmse_10cv)


## Stochastic Gradient Descent
sgdreg = SGDRegressor(penalty='l2', alpha=0.15, n_iter=200)
sgdreg.fit(X_train, y_train)
print('Stochastic Gradient Descent\nscore: ', sgdreg.score(X_train, y_train))

# Calculate RMSE
err = sgdreg.predict(X_train) - y_train
rmse_train = np.sqrt(np.dot(err,err) / len(y_train))

# Now let's compute RMSE using 10-fold x-validation
kf = KFold(n_splits=10)
xval_err = 0
for train, test in kf.split(X_train):
    #print(train, test)
    sgdreg.fit(X_train[train], y_train[train])
    p = sgdreg.predict(X_train[test])
    e = p - y_train[test]
    xval_err += np.dot(e,e)

rmse_10cv = np.sqrt(xval_err / len(X_train))
print('RMSE on training: %.4f' % rmse_train)
print('RMSE on 10-fold CV: %.4f' % rmse_10cv)


#-------------------------------
# Plotting prediction vs truth
#-------------------------------

fig, axarr = plt.subplots(1, 2, figsize=(12.8, 4.8))

axarr[0].plot(linreg.predict(X_train), y_train, 'b.', label='training')
axarr[0].plot(linreg.predict(X_test), y_test, 'ro', label='testing', markerfacecolor='w')
axarr[0].plot([-3,3], [-3,3], 'g-', label='ideal')

axarr[1].plot(sgdreg.predict(X_train), y_train, 'b.', label='training')
axarr[1].plot(sgdreg.predict(X_test), y_test, 'ro', label='testing', markerfacecolor='w')
axarr[1].plot([-3,3], [-3,3], 'g-', label='ideal')

# Now add the legend with some customizations.
axarr[0].legend(loc='upper left', borderaxespad=0., numpoints=1, scatterpoints=1, framealpha=1, edgecolor='k')
axarr[1].legend(loc='upper left', borderaxespad=0., numpoints=1, scatterpoints=1, framealpha=1, edgecolor='k')

axarr[0].set_title('Linear regression')
axarr[1].set_title('Stochastic gradient descent')

axarr[0].set_xlabel('Predicted')
axarr[0].set_ylabel('Real')
axarr[1].set_xlabel('Predicted')
axarr[1].set_ylabel('Real')

plt.savefig('regression_comparison.pdf', transparent=True, bbox_inches='tight')
plt.savefig('regression_comparison.png', transparent=True, bbox_inches='tight')
#plt.show()
fig.clf()
plt.clf()
plt.close('all')


#-------------------------------
# Make prediction for all data points
#-------------------------------

# Make prediction for all data
linreg.fit(X_train, y_train)
Y_predicted = linreg.predict(X_scaled)

# Scale the predicted result back to the original death rate
Y_predicted_rescaled_Array = scalerY.inverse_transform(Y_predicted)

data_complete_df['pre_deathRate'] = Y_predicted_rescaled_Array


# Calculate the number of predicted death from predicted death rate
data_complete_df['pre_death'] = data_complete_df.pre_deathRate * data_complete_df.population
data_complete_df['pre_death'] = data_complete_df['pre_death'].astype(int)


# Extract the predicted and true death in 2013 for making maps
data_df_4 = data_complete_df.loc[ data_complete_df.year == 2013,['state', 'year', 'pre_death', 'death']].copy()

data_df_4.to_csv('results.csv', index=False)



