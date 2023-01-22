#!/usr/bin/env python
# coding: utf-8

# ## Random Forest: Hyperparameters
# 
# Import [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) from `sklearn` and explore the hyperparameters.

# ### Import Random Forest Algorithm for Classification & Regression

# In[3]:


from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

RandomForestClassifier().get_params()


# The n_estimators hyperparameter is really straightforward, It simply controls how many individual decision trees are built and the max_depth hyperparameter controls how deep each of those individual decision trees can go. So in other words, a decision tree can keep splitting and splitting until it basically has a node that represents every given example in the training set. By setting max_depth, it constraints how many levels of splits it can make. In other words, it controls the complexity of the model and determines how closely it will fit to the training data.

# ![rf1.png](attachment:rf1.png)

# In[5]:


RandomForestRegressor().get_params()

