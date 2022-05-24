#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


# In[2]:


bank_data=pd.read_csv('bank-full.csv',sep=';')
bank_data


# In[3]:


bank_data.describe()


# In[4]:


bank_data.info()


# In[5]:


bank_data[bank_data.duplicated()]


# In[6]:


bank_data.job.value_counts()


# In[7]:


bank_data.marital.value_counts()


# In[8]:


bank_data.education.value_counts()


# In[9]:


bank_data.default.value_counts()


# In[10]:


bank_data.housing.value_counts()


# In[11]:


bank_data.loan.value_counts()


# In[12]:


bank_data.replace({"job":{"blue-collar":0,"management":1,"technician":2,"admin.":3,"services":4,"retired":5,"self-employed":6,"entrepreneur":7,"unemployed":8,"housemaid":9,"student":10,"unknown":11 }},inplace=True)
bank_data.replace({"marital":{"married":0,"single":1,"divorced":2}},inplace=True)
bank_data.replace({"education":{"unknown":0,"primary":1,"secondary":2,"tertiary":3}},inplace=True)
bank_data.replace({"default":{"no":0,"yes":1}},inplace=True)
bank_data.replace({"housing":{"no":0,"yes":1}},inplace=True)
bank_data.replace({"loan":{"no":0,"yes":1}},inplace=True)
bank_data.replace({"y":{"no":0,"yes":1}},inplace=True)
bank_data


# In[19]:


corr=bank_data.corr()
plt.figure(figsize=(10,6))
plt.title("Correlation")
sns.heatmap(corr,cmap="Blues")


# In[20]:


bank_data.info()


# In[21]:


bank_data=pd.get_dummies(bank_data,columns=['contact','poutcome','month'])
pd.set_option("display.max.columns", None)
bank_data


# In[22]:


sns.distplot(bank_data['age']);


# In[24]:


X=bank_data.drop(['y'],axis=1)
Y=bank_data['y']

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=4)

model=LogisticRegression()
model.fit(X_train,Y_train)


# In[25]:


trainp=model.predict(X_train)
accuracy = accuracy_score(trainp,Y_train)
print("The accuracy of model on training dataset is {}".format(accuracy))


# In[26]:


testp=model.predict(X_test)


# In[27]:


accuracy_of_test_data = accuracy_score(Y_test,testp)
print("The accuracy of model on testing dataset is {}".format(accuracy_of_test_data ))


# In[28]:


XP=model.predict(X)


# In[29]:


from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(Y,XP)
print(confusion_matrix)


# In[30]:


((39096 + 1235)/(39096 +826+4054+1235))*100


# In[34]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, model.predict_proba (X)[:,1])

auc = roc_auc_score(Y, XP)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='purple', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# In[35]:


print(auc)


# In[ ]:




