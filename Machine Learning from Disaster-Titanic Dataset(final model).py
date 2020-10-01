#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[7]:


# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# # Section 1: Data Exploration
# Step one, load the data.

# In[8]:


train = pd.read_csv("F:/DATA REPO/Datasets/titanic/titanic-train.csv")
test = pd.read_csv("F:/DATA REPO/Datasets/titanic/titanic-test.csv")


# In[9]:


train.info()


# In[10]:


train.describe()


# In[11]:


train.head(10)


# We can already see that we have some missing data in the Cabin variable. Let's have a look at what else we are missing.

# In[12]:


train.isnull().sum()


# Pretty good for the most part. Age is missing about 177 of its data, hopefully we will be able to use other data to provide a fair guess as to what those ages should be.
# Cabin is missing most of its data(687), but we might be able to learn something from the information that we have.
# Embarked is missing just two values, that won't be a problem.
# 
# Now, we canlook at the data via some graphs.

# # 2.0 Visual Exploratory Data Analysis (EDA) 
# And Your First Model

# In[13]:


sns.countplot(x='Survived', data=train);


# less people survived than didn't. Let's then build a first model that predicts that nobody survived.

# In[14]:


test['Survived'] = 0
test[['PassengerId', 'Survived']].to_csv('F:/DATA REPO/Datasets/titanic/no-survivours.csv', index=False)


# # EDA on Feature Variables

# In[16]:


sns.countplot(x='Pclass', data=train);


# It seems like you dont want to be in 3rd class...chances of surviving in 3rd class is limited ...okay!!! 
# can we now see what gender has to say about survival rate

# In[17]:


sns.countplot(x='Sex', data=train);


# ahhhhh! its so sad to be  a man. boychild is not safe ata all

# In[20]:


sns.factorplot(x='Survived', col='Sex', kind='count', data=train);


#  Furthermore, we can see that Women were more likely to survive than men.
#   Can we figure out how many women and how many men survived:

# In[21]:


train.groupby(['Sex']).Survived.sum()


#  What more??? we can figure out the proportion of women that survived, along with the proportion of men:

# In[22]:


print(train[train.Sex == 'female'].Survived.sum()/train[train.Sex == 'female'].Survived.count())
print(train[train.Sex == 'male'].Survived.sum()/train[train.Sex == 'male'].Survived.count())


# # 74% of women survived, while only 19% of men survived.it seems unlucky to be a man ahh!!

# In[24]:


test['Survived'] = test.Sex == 'female'
test['Survived'] = test.Survived.apply(lambda x: int(x))
test.head(10)


# In[25]:


test[['PassengerId', 'Survived']].to_csv('F:/DATA REPO/Datasets/titanic/women_survive.csv', index=False)


# # This model predict that all women survived and all men didn't. sounds unrealistic ahhh!
# Explore Our Data More!

# In[26]:


sns.factorplot(x='Survived', col='Pclass', kind='count', data=train);


# Passengers that travelled in first class were more likely to survive. On the other hand, passengers travelling in third class were more unlikely to survive.

# In[27]:


sns.factorplot(x='Survived', col='Embarked', kind='count', data=train);


# Passengers that embarked in Southampton(S) were less likely to survive.

# # EDA with Numeric Variables

# In[28]:


sns.distplot(train.Fare, kde=False);


# Most passengers paid less than USD.100 for travelling with the Titanic train

# In[29]:


train.groupby('Survived').Fare.hist(alpha=0.6);


# It looks as though those that paid more had a higher chance of surviving.Money matters ahh!!

# In[32]:


train_drop =train.dropna()
sns.distplot(train_drop.Age, kde=False);


# In[34]:


sns.stripplot(x='Survived', y='Fare', data=train, alpha=0.3, jitter=True);#stripplot


# In[35]:


sns.swarmplot(x='Survived', y='Fare', data=train);#swarmplot


# In[36]:


train.groupby('Survived').Fare.describe()


# lets see a scatter plot of Age against Fare

# In[40]:


sns.lmplot(x='Age', y='Fare', hue='Survived', data=train, fit_reg=False, scatter_kws={'alpha':0.9});


# It looks like those who survived either paid quite a bit for their ticket or they were young.

# In[41]:


sns.pairplot(train_drop, hue='Survived');


# In[44]:


train.Survived.plot(kind='hist', bins = 2, edgecolor = 'white')
plt.xticks((1, 0))
plt.xlabel(('Died','Survived'))
plt.show()
train.Survived.value_counts()


# more than half the passangers died

# # So far, so good, we have 
# explored our target variable visually and made your first predictions.
# explored some of our feature variables visually and made more predictions that did better based on our EDA.
# done some serious EDA of feature variables, categorical and numeric. Will now Build an ML model based on our EDA

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




