#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[2]:


# loadin the dataset
df = pd.read_csv('CarPricesPrediction.csv')


# In[3]:


# Summarizing the dataset
df


# In[5]:


df.info()


# In[8]:


len(df)


# In[9]:


df.describe()


# In[12]:


df.dtypes


# In[13]:


# pre-processing the dataset
# checking the null
df.isnull().sum()


# In[17]:


df.isnull().sum().max


# In[18]:


# we confirm that there are no null values in this dataset.


# In[21]:


# we have a duplicate column "Unnamed: 0" with serial numbering for the entries in the dataset.
df['Unnamed: 0'].unique


# In[27]:


# we have to drop this column
df.drop('Unnamed: 0', axis=1, inplace=True)


# In[28]:


df


# In[30]:


# the above is our cleaned dataset.


# In[ ]:


# There are 3 columns which are in "object" data types, so we will hve to perform label encoding in this dataset


# In[31]:


# we import additional libraries
from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[32]:


label = LabelEncoder()
label


# In[33]:


# List of columns to encode
columns_to_encode = ['Make','Model','Condition']


# In[34]:


# Apply label encoding to each column to be encoded
df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[35]:


#this is the final dataset with all integer values
df.head(5)


# In[36]:


df.dtypes


# In[37]:


# Data Visualisation of the above dataset


# In[59]:


df


# In[54]:


plt.scatter(df['Mileage'],df['Price'])
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Mileage vs Price')
plt.show()


# In[55]:


plt.scatter(df['Make'],df['Price'])
plt.xlabel('Make')
plt.ylabel('Price')
plt.title('Make vs Price')
plt.show()


# In[56]:


plt.scatter(df['Condition'],df['Price'])
plt.xlabel('Condition')
plt.ylabel('Price')
plt.title('Condition vs Price')
plt.show()


# In[93]:


plt.scatter(df['Year'],df['Price'])
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Year vs Price')
plt.show()


# In[97]:


plt.scatter(df['Model'],df['Price'])
plt.xlabel('Model')
plt.ylabel('Price')
plt.title('Model vs Price')
plt.show()


# In[104]:


# Calculate correlation matrix
corr = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[113]:


# We can confirm that Mileage and Year is directly proportional to the price of the car.


# In[114]:


#Segregating the Dataset into Input(x) and Output(y)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[115]:


x.shape,y.shape


# In[116]:


# Splitting the Dataset into Training and Testing Data
# we import additional libraries
from sklearn.model_selection import train_test_split
print('importing is done')


# In[117]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[118]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[119]:


# Loading the Models
# we import additional libraries
from sklearn.linear_model import Lasso, Ridge
# Lasso - L1
# Ridge - L2
print('importing is done')


# In[120]:


L1_model = Lasso()
L1_model


# In[121]:


L2_model = Ridge()
L2_model


# In[122]:


# Training the Models
L1 = L1_model.fit(x_train, y_train)
L2 = L2_model.fit(x_train, y_train)


# In[126]:


# Predicting the Result Using the Trained Models
y1_pred = L1.predict(x_test)
y1_pred


# In[124]:


y2_pred = L2.predict(x_test)
y2_pred


# In[127]:


# Calculating the Accuracy of the Trained Models
# we import additional library for accuracy
from sklearn.metrics import r2_score
print('importing is done')


# In[128]:


LR1 = r2_score(y_test, y1_pred)
LR1


# In[129]:


LR2 = r2_score(y_test, y2_pred)
LR2


# In[130]:


# Predicting the Output of Single Test Data using the Trained Model
y_test


# In[131]:


x_test[4]


# In[132]:


x_test[4].shape


# In[133]:


x_test[4].reshape(1,5)


# In[134]:


x_test[4].reshape(1,5).shape


# In[135]:


L1.predict(x_test[4].reshape(1,5))


# In[136]:


L2.predict(x_test[4].reshape(1,5))


# In[137]:


y_test[4]


# In[138]:


# Visualizing the Result
# L1 Model


# In[139]:


plt.scatter(y_test,y1_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])


# In[140]:


# L2 Model


# In[141]:


plt.scatter(y_test,y2_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])


# In[142]:


# Conclusion: The line of regression passes in such a way that it shows direct co-relation of mileage and price.


# In[ ]:




