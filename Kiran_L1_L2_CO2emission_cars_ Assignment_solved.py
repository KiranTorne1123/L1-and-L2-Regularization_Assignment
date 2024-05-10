#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[3]:


#loading the dataset
df=pd.read_csv('data.csv')


# In[4]:


# summarizing the dataset
df


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.dtypes


# In[8]:


df['Unnamed: 5'].unique


# In[9]:


# pre-processing the dataset
# check the null values
df.isnull().sum()


# In[10]:


# treat the null values
# Drop columns with any null values
df.dropna(axis=1, inplace=True)


# In[11]:


# this is the cleaned dataset
df


# In[12]:


# confirming further any other possible null values
df.isnull().sum().max()


# In[13]:


df.info()


# In[14]:


# wwe have 2 columns with object values which we have to convert to integer


# In[15]:


# we import additional libraries
from sklearn.preprocessing import LabelEncoder
print('importing is done')


# In[16]:


label = LabelEncoder()
label


# In[17]:


# List of columns to encode
columns_to_encode = ['Car','Model']


# In[18]:


# Apply label encoding to each column to be encoded
df[columns_to_encode]=df[columns_to_encode].apply(LabelEncoder().fit_transform)


# In[19]:


#this is the final dataset with all integer values
df.head(5)


# In[20]:


# confirming that all are int types
df.info()


# In[21]:


# visualising the dataset
df.head()


# In[22]:


# using scatter plot
plt.scatter(df['Car'],df['CO2'])
plt.xlabel('Car')
plt.ylabel('CO2')
plt.title('Car vs CO2')
plt.show()


# In[23]:


plt.scatter(df['Model'],df['CO2'])
plt.xlabel('Model')
plt.ylabel('CO2')
plt.title('Model vs CO2')
plt.show()


# In[24]:


plt.scatter(df['Volume'],df['CO2'])
plt.xlabel('Volume')
plt.ylabel('CO2')
plt.title('Volume vs CO2')
plt.show()


# In[25]:


plt.scatter(df['Weight'],df['CO2'])
plt.xlabel('Weight')
plt.ylabel('CO2')
plt.title('Weight vs CO2')
plt.show()


# In[26]:


plt.scatter(df['Car'],df['CO2'], label = 'Car')
plt.scatter(df['Model'],df['CO2'], label = 'Model')
plt.scatter(df['Volume'],df['CO2'], label = 'Volume')
plt.scatter(df['Weight'],df['CO2'], label = 'Weight')
plt.xlabel('Car_Model_Volume_Weight')
plt.ylabel('CO2')
plt.title('Car_Model_Volume_Weight vs CO2 Emissions')
plt.legend()
plt.show()


# In[27]:


# Inference from the chart above.
# 1) The CO2 emssions are directly proportional to weight of the car and volume of the engine.
# 2) There are other varibles as well leading to excessive CO2 emissions such as, type os engine (petrol or diesel), mileage driven, maintainenece of engine etc
# 3)Air to Fuel ratio for larger engines is more as a result more CO2 emissions are caused (Ideal ratio is 14.7:1)


# In[28]:


# Using Bar chart
plt.bar(df['Car'],df['CO2'], label = 'Car')
plt.bar(df['Model'],df['CO2'], label = 'Model')
plt.bar(df['Volume'],df['CO2'], color = 'k', label = 'Volume')
plt.bar(df['Weight'],df['CO2'], label = 'Weight')
plt.xlabel('Car_Model_Volume_Weight')
plt.ylabel('CO2')
plt.title('Car_Model_Volume_Weight vs CO2 Emissions')
plt.legend()
plt.show()


# In[29]:


#Segregating the Dataset into Input(x) and Output(y)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[30]:


x.shape


# In[31]:


y.shape


# In[32]:


# Splitting the Dataset into Training and Testing Data
# we import additional libraries
from sklearn.model_selection import train_test_split
print('importing is done')


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[34]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[35]:


# Loading the Models
# we import additional libraries
from sklearn.linear_model import Lasso, Ridge
# Lasso - L1
# Ridge - L2
print('importing is done')


# In[36]:


L1_model = Lasso()
L1_model


# In[37]:


L2_model = Ridge()
L2_model


# In[38]:


# Training the Models
L1 = L1_model.fit(x_train, y_train)


# In[39]:


L2 = L2_model.fit(x_train, y_train)


# In[40]:


# Predicting the Result Using the Trained Models
y1_pred = L1.predict(x_test)
y1_pred


# In[41]:


y2_pred = L2.predict(x_test)
y2_pred


# In[42]:


# Calculating the Accuracy of the Trained Models
# we import additional library for accuracy
from sklearn.metrics import r2_score
print('importing is done')


# In[43]:


LR1 = r2_score(y_test, y1_pred)
LR1


# In[44]:


LR2 = r2_score(y_test, y2_pred)
LR2


# In[45]:


# Predicting the Output of Single Test Data using the Trained Model
y_test


# In[46]:


x_test[3]


# In[47]:


x_test[3].shape


# In[48]:


x_test[3].reshape(1,4)


# In[49]:


x_test[3].reshape(1,4).shape


# In[50]:


L1.predict(x_test[3].reshape(1,4))


# In[51]:


L2.predict(x_test[3].reshape(1,4))


# In[52]:


y_test[3]


# In[53]:


# Visualizing the Result
# L1 Model


# In[54]:


plt.scatter(y_test,y1_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])


# In[55]:


# L2 Model


# In[56]:


plt.scatter(y_test,y2_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])


# In[57]:


# Conclusion: The line of regression passes in such a way that it is near to all the points available on the scatter plot


# In[ ]:





# In[ ]:




