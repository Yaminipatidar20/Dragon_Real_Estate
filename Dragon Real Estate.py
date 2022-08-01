#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate - Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
# In[8]:


# for plotting histrogram
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))


# ## Train - test splitting

# In[9]:


# for learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


# train_set, test_set = split_train_test(housing, 0.2)


# In[11]:


# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set['CHAS'].value_counts()


# In[15]:


strat_train_set['CHAS'].value_counts()


# In[16]:


# 95/7


# In[17]:


# 376/28


# In[18]:


housing = strat_train_set.copy()


# ## Looking for Correlations

# In[19]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM","ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12, 8))


# In[20]:


housing.plot(kind="scatter", x="RM", y="MEDV",alpha=0.8)


# ## Trying out Attribute Combination

# In[21]:


housing["TAXRM"] = housing['TAX']/housing['RM']


# In[22]:


housing.head()                                         #housing["TAXRM"]


# In[23]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[24]:


housing.plot(kind="scatter", x="RM", y="MEDV",alpha=0.8)


# In[25]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing Attributes

# To take care of missing attributes, you have three option:-
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)

# In[26]:


a = housing.dropna(subset=["RM"])    # option 1
a.shape
# Note that the original housing dataframe will remain unchnaged 


# In[27]:


housing.drop("RM", axis=1).shape           # option 2
# Note that there is no RM column ad also note that the original housing dataframe will remain unchnaged 


# In[28]:


median = housing["RM"].median()           # compute median for option 3


# In[29]:


housing["RM"].fillna(median)
# Note that the original housing dataframe will remain unchnaged 


# In[30]:


housing.shape


# In[31]:


housing.describe()     # before we started filling missing attibutes


# In[32]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[33]:


imputer.statistics_


# In[34]:


X = imputer.transform(housing)


# In[35]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[36]:


housing_tr.describe()


# ## Scikit-learn Design

# Primarily , three types of objects
# 1. Estimators - It estimates some parameter based on dataset,  eg:- imputer   It has a fit method and transform method.                                    Fit method - fits the dataset and calculate internal parameters
# 
# 2. Transformers - Transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which filts and then transform.
# 
# 3. Predictors - LinearRegression mpdel is an example of predictors. fit() and predict() are two common functions. It also gives score() function which will evaluate the perdictions.

# ## Feature Scalling

# Primarily, two types of feature scalling method:
# 1. Min-max scalling(Normalization)   
#    (value-min)/(max-min)
#    sklearn provides a class called MinMaxScaler for this
#    
# 2. Standardization
#    (value - min)/std
#    sklearn provides a class called StandardScaler for this 

# ## Creating a Pipeline

# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
#    ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[38]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[39]:


housing_num_tr.shape


# ## Selecting a desired model for Dragon Real Estates

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[41]:


some_data = housing.iloc[:5]


# In[42]:


some_labels = housing_labels.iloc[:5]


# In[43]:


prepared_data = my_pipeline.transform(some_data)


# In[44]:


model.predict(prepared_data)


# In[45]:


list(some_labels)


# ## Evaluating the model

# In[46]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[47]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[48]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[49]:


rmse_scores


# In[50]:


def print_scores(scores):
    print("Scores is:-", scores)
    print("Mean is:-", scores.mean())
    print("Standard deviation:-", scores.std())


# In[51]:


print_scores(rmse_scores)


# Quiz : Convert this notebook into a python file and run the pipeline using VS Code 

# ## Saving the Model

# In[53]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# ## Testing the model on test data

# In[58]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[55]:


final_rmse


# In[60]:


prepared_data[0]


# ## Using the Model

# In[61]:


from joblib import dump, load
import numpy as np
model = load('dragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)


# In[ ]:




