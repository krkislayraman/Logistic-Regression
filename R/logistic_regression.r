# Install the necessary packages(required only once)
install.packages("ROCR")
install.packages("car")
install.packages("DescTools")

# Load the packages/libraries
library(car)
library(ROCR)
library(DescTools)

# Set Working Directory
setwd("C:/.../Documents/R")

# Read the data files
TrainRaw = read.csv("R_Module_Day_7.2_Credit_Risk_Train_data.csv",na.strings = "") # na.strings - Account for NAs

TestRaw = read.csv("R_Module_Day_8.2_Credit_Risk_Test_data.csv", na.strings = "") # na.strings - Account for NAs

# Create a new column. Call it "Source". Assign the value of "Train" and "Test" in that column respectively
TrainRaw$Source = "Train"
TestRaw$Source = "Test"

# Combine both datasets. Call it FullRaw
FullRaw = rbind(TrainRaw,TestRaw)
dim(FullRaw)

# View the data
View(FullRaw)

# Check for NAs
colSums(is.na(FullRaw)) # There are NA values in more than 3 columns

# Check the summary of the data
summary(FullRaw) 

str(FullRaw)

#############################################
# Missing Value Imputation
#############################################

# Imputation of missing values in Categorical Variable
# Look for the "mode" (most frequent category) in every categorical variable and 
# use that to replace NA values in that variable

# Bear in mind that the "most frequent category" or "mode" of a categorical
# variable needs to found from the TRAIN dataset and NOT FULL dataset

summary(FullRaw) 
# LoanAmount has outliers on the higher side
# Loan_Amount_Term has outliers on the lower side
# Credit_History is a binary variable. You can use the median to impute NAs

# Imputaion of Continuous data
MyColumnName = "LoanAmount"
MyMedian = median(FullRaw[FullRaw$Source == "Train", MyColumnName], na.rm = TRUE) # Find median from train rows
MissingValueRows = is.na(FullRaw[,MyColumnName]) # Find missing value rows for the column
FullRaw[MissingValueRows,MyColumnName] = MyMedian # Impute NA values with MyMedian
summary(FullRaw) # Validate

MyColumnName = "Credit_History"
MyMedian = median(FullRaw[FullRaw$Source == "Train", MyColumnName], na.rm = TRUE) # Find median from train rows
MissingValueRows = is.na(FullRaw[,MyColumnName]) # Find missing value rows for the column
FullRaw[MissingValueRows,MyColumnName] = MyMedian # Impute NA values with MyMedian
summary(FullRaw) # Validate

colSums(is.na(FullRaw))

MyColumnName = "Loan_Amount_Term"
MyMedian = median(FullRaw[FullRaw$Source == "Train", MyColumnName], na.rm = TRUE) # Find median from train rows
MissingValueRows = is.na(FullRaw[,MyColumnName]) # Find missing value rows for the column
FullRaw[MissingValueRows,MyColumnName] = MyMedian # Impute NA values with MyMedian
summary(FullRaw) # Validate

# Imputation for categorical data
MyColumnName = "Gender"
MyMode = Mode(FullRaw[FullRaw$Source == "Train", MyColumnName], na.rm = TRUE) # Find median from train rows
MissingValueRows = is.na(FullRaw[,MyColumnName]) # Find missing value rows for the column
FullRaw[MissingValueRows,MyColumnName] = MyMode # Impute NA values with MyMedian
summary(FullRaw) # Validate

MyColumnName = "Self_Employed"
MyMode = Mode(FullRaw[FullRaw$Source == "Train", MyColumnName], na.rm = TRUE) # Find median from train rows
MissingValueRows = is.na(FullRaw[,MyColumnName]) # Find missing value rows for the column
FullRaw[MissingValueRows,MyColumnName] = MyMode # Impute NA values with MyMedian
summary(FullRaw) # Validate

MyColumnName = "Dependents"
MyMode = Mode(FullRaw[FullRaw$Source == "Train", MyColumnName], na.rm = TRUE) # Find median from train rows
MissingValueRows = is.na(FullRaw[,MyColumnName]) # Find missing value rows for the column
FullRaw[MissingValueRows,MyColumnName] = MyMode # Impute NA values with MyMedian
summary(FullRaw) # Validate

MyColumnName = "Married"
MyMode = Mode(FullRaw[FullRaw$Source == "Train", MyColumnName], na.rm = TRUE) # Find median from train rows
MissingValueRows = is.na(FullRaw[,MyColumnName]) # Find missing value rows for the column
FullRaw[MissingValueRows,MyColumnName] = MyMode # Impute NA values with MyMedian
summary(FullRaw) # Validate

# Create dummy dataframe for dummy variables
Dummy_df = model.matrix(~Gender+Married+Dependents+Education+Self_Employed+Property_Area, data = FullRaw)
View(Dummy_Df)

Dummy_Df = Dummy_df[,-c(1)]

# Drop the columns of which dummy variables were created
FullRaw <- subset(FullRaw, select = -c(Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_ID))

# COmbine the FullRaw and Dummy_Df dataframe
Data = cbind(FullRaw, Dummy_Df)
dim(Data)

Data$Loan_Status = ifelse(Data$Loan_Status == "N", 1, 0)
str(Data)

# Smapling the data into Test and Train
Train_Data = subset(Data, Source == "Train")
summary(Train_Data)
str(Train_Data)
dim(Train_Data)
Train_Data = subset(Train_Data, select = -c(Source))

Test_Data = subset(Data, Source == "Test")
Test_Data = subset(Test_Data, select = -c(Source))
dim(Test_Data)

# Build logistic regression model, generalised linear model
M1 = glm(Loan_Status ~ ., data = Train_Data, family = "binomial")
summary(M1)

# Updation of model based on independent variables' significance
M2 = update(M1, .~. - `Dependents3+`)
summary(M2)
M3 = update(M2, .~. - Self_EmployedYes)
summary(M3)
M4 = update(M3, .~. - GenderMale)
summary(M4)
M5 = update(M4, .~. - ApplicantIncome)
summary(M5)
M6 = update(M5, .~. - Loan_Amount_Term)
summary(M6)
M7 = update(M6, .~. - Dependents2)
summary(M7)
M8 = update(M7, .~. - Property_AreaUrban)
summary(M8)
M9 = update(M8, .~. - LoanAmount)
summary(M9)
M10 = update(M9, .~. - `EducationNot Graduate`)
summary(M10)

# Predict on Testset using final model
Test_Prob = predict(M10, Test_Data, type = "response") # type = "response" gives a probability values
head(Test_Prob)
tail(Test_Prob)

# Classify the test predictions into classes of 0s and 1s
Test_Class = ifelse(Test_Prob >= 0.5, 1, 0)
head(Test_Class)

# Confusion Matrix
table(Test_Class, Test_Data$Loan_Status)

# ROC Curve
ROC_Pred_Test <- prediction(Test_Prob, Test_Data$Loan_Status)

ROC_Curve <- performance(ROC_Pred_Test, "tpr", "fpr")

# Plot the ROC_Curve
plot(ROC_Curve)
