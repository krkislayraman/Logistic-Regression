# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:44:30 2019

@author: RAMAN
"""

import os
import numpy as np
import pandas as pd

os.chdir("C:\\Users\\RAMAN\\Documents\\Python Scripts\\Data")
os.getcwd()

# Read the data
train_data = pd.read_csv('R_Module_Day_7.2_Credit_Risk_Train_data.csv')
test_data = pd.read_csv('R_Module_Day_8.2_Credit_Risk_Test_data.csv')

# Add a Source column in the dataframe
train_data['Source'] = 'Train'
test_data['Source'] = 'Test'
raw_data = pd.concat([train_data, test_data], axis = 0)
raw_data.shape

# Check the summary of the data
summary_of_raw = raw_data.describe(include = 'all')
raw_data.dtypes

# Check for NA values in all the columns
raw_data.isnull().sum()

# Fill the NA values with mode and median for both categorical and continuous variables
for i in raw_data.columns:
    if(raw_data[i].dtype == object):
        temp_imputation_value = raw_data.loc[raw_data['Source'] == "Train", i].mode()[0]
        raw_data[i] = raw_data[i].fillna(temp_imputation_value)
        
    else:
        temp_imputation_value = raw_data.loc[raw_data['Source'] == "Train", i].median()
        raw_data[i].fillna(temp_imputation_value, inplace = True)

# Verify the result
raw_data.isnull().sum()

# creation of dummy variables
categorical_Variable = raw_data.loc[:, raw_data.dtypes == object].columns

categorical_Variable = raw_data.loc[:,raw_data.dtypes==object].columns
Dummy_Df = pd.get_dummies(raw_data[categorical_Variable].drop(['Loan_ID','Source','Loan_Status'],axis=1), drop_first = True, dtype = int)
Dummy_Df.columns

Full_Data = pd.concat([raw_data, Dummy_Df],axis=1)
Full_Data.dtypes

Cols_To_Drop = categorical_Variable.drop(['Source','Loan_Status'])

Full_Data = Full_Data.drop(Cols_To_Drop,axis=1).copy()

# Verify the changes
Full_Data.shape
Full_Data.columns

# Convert the Loan Status column into 1s and 0s
Full_Data['Loan_Status']=np.where(Full_Data['Loan_Status']== 'N',1,0)

full_raw_data = Full_Data.copy()

# Adding Intercept column to the dataframe
full_raw_data['Intercept'] = 1
full_raw_data.shape


# Sampling into Train_X, Train_Y, Test_X. Test_Y
Train_X = full_raw_data.loc[full_raw_data['Source'] == 'Train'].drop(['Source', 'Loan_Status'], axis = 1).copy()

Train_Y = full_raw_data.loc[full_raw_data['Source'] == 'Train']
Train_Y = Train_Y['Loan_Status'].copy()
Train_Y.shape

Test_X = full_raw_data.loc[full_raw_data['Source'] == 'Test'].drop(['Source', 'Loan_Status'], axis = 1).copy()

Test_Y = full_raw_data.loc[full_raw_data['Source'] == 'Test']
Test_Y = Test_Y['Loan_Status'].copy()
Test_Y.shape

###########################
# Model Building
###########################

# Build logistic regression model (using statsmodels package/library)
# And drop the insignificant variables

from statsmodels.api import Logit
M1 = Logit(Train_Y, Train_X) # (Dep_Var, Indep_Vars) # this is model definition
M1_Model = M1.fit() # This is model building
M1_Model.summary() # This is model output/summary

Cols_to_drop = ['Dependents_3+']
M2 = Logit(Train_Y, Train_X.drop(Cols_to_drop, axis = 1))
M2_Model = M2.fit()
M2_Model.summary()

Cols_to_drop.append('Self_Employed_Yes')

M3 = Logit(Train_Y, Train_X.drop(Cols_to_drop, axis = 1))
M3_Model = M3.fit()
M3_Model.summary()

Cols_to_drop.append('Gender_Male')
M4 = Logit(Train_Y, Train_X.drop(Cols_to_drop, axis = 1))
M4_Model = M4.fit()
M4_Model.summary()

Cols_to_drop.append('ApplicantIncome')
M5 = Logit(Train_Y, Train_X.drop(Cols_to_drop, axis = 1))
M5_Model = M5.fit()
M5_Model.summary()


Cols_to_drop.append('Loan_Amount_Term')
M6 = Logit(Train_Y, Train_X.drop(Cols_to_drop, axis = 1))
M6_Model = M6.fit()
M6_Model.summary()

Cols_to_drop.append('Dependents_2')
M7 = Logit(Train_Y, Train_X.drop(Cols_to_drop, axis = 1))
M7_Model = M7.fit()
M7_Model.summary()

Cols_to_drop.append('Property_Area_Urban')
M8 = Logit(Train_Y, Train_X.drop(Cols_to_drop, axis = 1))
M8_Model = M8.fit()
M8_Model.summary()

Cols_to_drop.append('LoanAmount')
M9 = Logit(Train_Y, Train_X.drop(Cols_to_drop, axis = 1))
M9_Model = M9.fit()
M9_Model.summary()

Cols_to_drop.append('Education_Not Graduate')
M10 = Logit(Train_Y, Train_X.drop(Cols_to_drop, axis = 1))
M10_Model = M10.fit()
M10_Model.summary()


###########################################
# Predict on testset considering 0.5 as a cut-off point
###########################################
Test_X = Test_X.drop(Cols_to_drop, axis = 1)

Test_X['Test_Prob'] = M10_Model.predict(Test_X)

Test_X.columns

Test_X['Test_Class'] = np.where(Test_X['Test_Prob'] >= 0.5, 1, 0)

###########################################
# Confusion Matrix
###########################################

Confusion_Mat = pd.crosstab(Test_X['Test_Class'], Test_Y) # R, C format
Confusion_Mat

#TPR = 58/77
#FPR = 1/289

# Check the accuracy of the model
Accuracy = sum(np.diagonal(Confusion_Mat))/Test_X.shape[0]*100            # 94.55040871934605

###########################################
# AUC and ROC Curve
###########################################

from sklearn.metrics import roc_curve, auc

# predict on train data
Train_X['Train_Prob'] = M10_Model.predict(Train_X.drop(Cols_to_drop, axis = 1))

# Calculate FPR, TPR and cut-off thresholds
fpr, tpr, thresholds = roc_curve(Train_Y, Train_X['Train_Prob'])

ROC_Df = pd.DataFrame()
ROC_Df['FPR'] = fpr
ROC_Df['TPR'] = tpr
ROC_Df['Cutoff'] = thresholds

# Plot the ROC Curve
import seaborn as sns
sns.lineplot(ROC_Df['FPR'], ROC_Df['TPR'])

# Area under the curve
auc(fpr, tpr)         # 0.7828482918641391

