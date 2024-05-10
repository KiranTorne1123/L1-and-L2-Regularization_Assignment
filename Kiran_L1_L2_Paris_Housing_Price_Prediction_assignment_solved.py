#!/usr/bin/env python
# coding: utf-8

# In[23]:


# import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
print('importing is done')


# In[24]:


# loadin the dataset
df = pd.read_csv('ParisHousing.csv')


# In[25]:


# Summarizing the dataset
df


# In[26]:


df.info()


# In[27]:


len(df)


# In[28]:


df.describe()


# In[29]:


df.dtypes


# In[30]:


# pre-processing the dataset
# checking the null
df.isnull().sum()


# In[31]:


df.isnull().sum().max


# In[32]:


# it is confirmed that we dont have any null values


# In[33]:


# Since all the values are numeric we dont need to do any label encoding in this dataset


# In[34]:


# visualising the dataset
df.head()


# In[35]:


# Data Visualisation of the above dataset


# In[36]:


plt.scatter(df['squareMeters'],df['price'])
plt.xlabel('squareMeters')
plt.ylabel('price')
plt.title('SquareMeters vs price')
plt.show()


# In[37]:


plt.scatter(df['squareMeters'],df['price'], label = 'squareMeters')
plt.scatter(df['numberOfRooms'],df['price'], label = 'numberOfRooms')
plt.scatter(df['hasYard'],df['price'], label = 'hasYard')
plt.scatter(df['hasPool'],df['price'], label = 'hasPool')
plt.scatter(df['floors'],df['price'], label = 'floors')
plt.scatter(df['cityCode'],df['price'], label = 'cityCode')
plt.scatter(df['cityPartRange'],df['price'], label = 'cityPartRange')
plt.scatter(df['numPrevOwners'],df['price'], label = 'numPrevOwners')
plt.scatter(df['made'],df['price'], label = 'made')
plt.scatter(df['isNewBuilt'],df['price'], label = 'isNewBuilt')
plt.scatter(df['hasStormProtector'],df['price'], label = 'hasStormProtector')
plt.scatter(df['basement'],df['price'], label = 'basement')
plt.scatter(df['attic'],df['price'], label = 'attic')
plt.scatter(df['garage'],df['price'], label = 'garage')
plt.scatter(df['hasStorageRoom'],df['price'], label = 'hasStorageRoom')
plt.scatter(df['hasGuestRoom'],df['price'], label = 'hasGuestRoom')
plt.xlabel('Paris_Housing')
plt.ylabel('Price')
plt.title('Paris_Housing vs Price')
plt.legend()
plt.show()


# In[38]:


# Calculate correlation matrix
corr = df.corr()


# In[39]:


# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[40]:


# From the above data visualisation, we can confirm that there is a direct co-relation of Area in Sqare metres and Price.


# In[ ]:


#Segregating the Dataset into Input(x) and Output(y)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[41]:


x.shape


# In[43]:


y.shape


# In[44]:


# Splitting the Dataset into Training and Testing Data
# we import additional libraries
from sklearn.model_selection import train_test_split
print('importing is done')


# In[45]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[46]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[47]:


# Loading the Models
# we import additional libraries
from sklearn.linear_model import Lasso, Ridge
# Lasso - L1
# Ridge - L2
print('importing is done')


# In[48]:


L1_model = Lasso()
L1_model


# In[49]:


L2_model = Ridge()
L2_model


# In[50]:


# Training the Models
L1 = L1_model.fit(x_train, y_train)


# In[51]:


L2 = L2_model.fit(x_train, y_train)


# In[52]:


# Predicting the Result Using the Trained Models
y1_pred = L1.predict(x_test)
y1_pred


# In[53]:


y2_pred = L2.predict(x_test)
y2_pred


# In[54]:


# Calculating the Accuracy of the Trained Models
# we import additional library for accuracy
from sklearn.metrics import r2_score
print('importing is done')


# In[55]:


LR1 = r2_score(y_test, y1_pred)
LR1


# In[56]:


LR2 = r2_score(y_test, y2_pred)
LR2


# In[57]:


# Predicting the Output of Single Test Data using the Trained Model
y_test


# In[58]:


x_test[2]


# In[61]:


x_test[2].shape


# In[62]:


x_test[2].reshape(1,16)


# In[63]:


x_test[2].reshape(1,16).shape


# In[64]:


L1.predict(x_test[2].reshape(1,16))


# In[65]:


L2.predict(x_test[2].reshape(1,16))


# In[66]:


y_test[2]


# In[ ]:


# Visualizing the Result
# L1 Model


# In[67]:


plt.scatter(y_test,y1_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])


# In[ ]:


# L2 Model


# In[68]:


plt.scatter(y_test,y2_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])


# In[ ]:


# Conclusion: The line of regression passes in such a way that it shows direct co-relation betweeen Area (sq.mt) & Price.


# In[ ]:





# In[ ]:




