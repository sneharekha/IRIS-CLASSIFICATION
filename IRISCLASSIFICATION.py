#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df=pd.read_csv("C:/Users/gupta/OneDrive/Desktop/csv/train.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.columns


# In[6]:


print(df.isnull().sum())


# In[7]:


df.info()


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


X=df[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType']]
y=df['SalePrice']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)


# In[ ]:


X_train


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()

# Fitting the model on the training data
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)

# Mean Squared Error and R-squared for model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[ ]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()


# In[ ]:


residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




