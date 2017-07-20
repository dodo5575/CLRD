# CLRD
Predicting the chronic lower respiratory diseases (CLRD) death rate with machine learning 

# Major files

├── README.md    
├── data    
│   ├── PM25    
│   │   ├── raw data ...    
│   │   ├── clean.csv    
│   │   └── clean.py    
│   ├── dataSource.txt    
│   ├── death    
│   │   ├── raw data ...    
│   │   ├── clean.csv    
│   │   ├── clean.py    
│   ├── income    
│   │   ├── raw data ...    
│   │   ├── clean.csv    
│   │   └── clean.py    
│   ├── nHospital    
│   │   ├── raw data ...    
│   │   ├── clean.csv    
│   │   ├── clean.py    
│   ├── population    
│   │   ├── raw data ...    
│   │   ├── clean.csv    
│   │   ├── clean.py    
│   ├── smokingRate    
│   │   ├── raw data ...    
│   │   ├── clean.csv    
│   │   ├── clean.py    
│   └── temperature    
│       ├── raw data ...    
│       ├── clean.csv    
│       └── clean.py    
├── plotPredictionMap.py    
├── plotTruthMap.py    
├── regression.py    
├── results.csv    
└── usStates.py    


The 'data' folder contain 'dataSource.txt' and datasets in each folder. In each dataset folder, 'clean.py' is the script for cleaning the raw data and produce 'clean.csv'.

'regression.py': The main script for data analysis and regression. 
'plotPredictionMap.py' and 'plotTruthMap.py': Plotting the prediction and truth map.
'usStates.py': For the code-to-name conversion of states in the US.

