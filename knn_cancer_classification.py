# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:23:21 2020
#author: Nikita Porwal
#Email - nikitaporwal05@gmail.com
#LinkedIn - https://www.linkedin.com/in/nikita-porwal/
"""

#kNN

import numpy as np
import pandas as pd

#Impoting the Data
cancerdata = pd.read_csv(r"/Users/vaibhavi/Desktop/kNN/cancerdata.csv")

#Loading our custom functions

class Data_Auditor:
    def NA_in_Data(data_frame):
        result = (data_frame.isnull().sum().sum()) / (data_frame.shape[0] * data_frame.shape[1]) * 100
        return(print("Data has",round(result,2),"% NA's"))
    
    def Remove_Columns(data_frame,*args):
        list_of_cols = list(args)
        data_frame.drop(list_of_cols,axis = 1,inplace = True)
        
    def NA_in_Columns(data_frame):
        total_missing = data_frame.isnull().sum().sort_values(ascending=False)
        percent_missing = round(((data_frame.isnull().sum()/data_frame.isnull().count()).sort_values(ascending=False)*100),1)
        missing_data = pd.concat([total_missing, percent_missing], axis=1, keys=['Missing_Obs', 'Percent_of_NA'])
        return(missing_data.head(10))
        
    def Most_Frequent_Data(data_frame,Column):
        Count = data_frame[Column].value_counts()
        Percentage = round(((data_frame[Column].value_counts()/data_frame.shape[0])*100),2)
        Summary_data = pd.concat([Count, Percentage], axis=1, keys=['Count','Percentage'])
        return(Summary_data)
        
    def Remove_Outlier(data_frame,low,high):
        quant_df = data_frame.quantile([low, high])
        for cols in list(data_frame.columns):
            if data_frame[cols].dtypes == 'float64' or data_frame[cols].dtypes == 'int64':
                df = data_frame[(data_frame[cols] > quant_df.loc[low, cols]) & (data_frame[cols] < quant_df.loc[high, cols])]
                return(df)
                
    def Remove_Multicollinearity(data_frame,threshold):
        corr_matrix = data_frame.corr().abs() # Create correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) # Select upper triangle of correlation matrix
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]# Find features with correlation greater than threshold
        data_frame.drop(to_drop, axis=1, inplace=True)# Drop features
        print(to_drop,'is removed')



#Check for NA's
Data_Auditor.NA_in_Data(cancerdata)


#Lets look into the data
cancerdata.info()


#Remove usless columns
Data_Auditor.Remove_Columns(cancerdata,'id')


def diagnosis_value(diagnosis): 
	if diagnosis == 'M': 
		return 1
	else: 
		return 0

cancerdata['diagnosis'] = cancerdata['diagnosis'].apply(diagnosis_value)


#Splitting Data
X = np.array(cancerdata.iloc[:, 1:]) 
y = np.array(cancerdata['diagnosis'])
del cancerdata


#Splitting data to training and testing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size = 0.1, random_state = 0) 
del X
del y


#Using Sklearn
from sklearn.neighbors import KNeighborsClassifier

#Best starting point for k is the square root of train size
import math
k = int(round(math.sqrt(X_train.shape[0]),0))

knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean') 
knn.fit(X_train, y_train)

del k



#Predicted
y_pred = knn.predict(X_test)


#Confusion Matrix to check Accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

Accuracy = round((34 + 17)/(34 + 17 + 5 + 1),2)
Accuracy #We have 89% accuracy


#Lets see if we can better this using CV
neighbors = [] 
cv_scores = [] 
  
from sklearn.model_selection import cross_val_score 
# perform 10 fold cross validation 
for k in range(1, 40, 1): 
    neighbors.append(k) 
    knn = KNeighborsClassifier(n_neighbors = k) 
    scores = cross_val_score( 
        knn, X_train, y_train, cv = 10, scoring = 'accuracy') 
    cv_scores.append(scores.mean())


#Misclassification error versus k
import matplotlib.pyplot as plt 

MSE = [1-x for x in cv_scores] 
  
# determining the best k 
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is % d ' % optimal_k) 
  
# plot misclassification error versus k 
plt.figure(figsize = (10, 6)) 
plt.plot(neighbors, MSE) 
plt.xlabel('Number of neighbors') 
plt.ylabel('Misclassification Error') 
plt.show() 

#9 looks like a good fit lets try
knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean') 
knn.fit(X_train, y_train) 

#Predicted
y_pred = knn.predict(X_test)


#Confusion Matrix to check Accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

Accuracy = round((34 + 17)/(34 + 17 + 5 + 1),2)
Accuracy #We have 89% accuracy

