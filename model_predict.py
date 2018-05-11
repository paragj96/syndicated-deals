
# coding: utf-8

# # Predicting customers who will "charge-off"
# *produced by Vincenzo Pota in August 2017 *

# This notebook contains my attempt to predict customers who will charge-off in the future. I describe in detail the following steps:
# 1. Data Cleaning
# 2. Feature selection and transformation
# 3. Define the business case
# 4. Build the models
# 5. Test the models
# 
# Dataset is given in a flat file and in a database. Let's use the database for good practice and for performances. Once the dataset is better understood, we can perform data cleaning and aggregation in-database to avoid overloading computer memory. 

# Let's load the libraries, connect to the database, parse dates and load all data in-memory:

# In[1]:


import sqlite3
import pandas as pd 
import numpy as np 


conn = sqlite3.connect('/home/lenovo/projects/database.sqlite') # This might take a while to run...
to_parse = ['issue_d' , 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
df = pd.read_sql_query('select * from loan', con=conn, parse_dates = to_parse)


# The dataframe `df` has 887,384 rows and 75 columns. It occupies 0.5Gb, which might be problematic for later data modelling.

# ## Data Cleaning

# After a closer inspection in Excel, many of the columns seem to contain very little information. I will remove these columns to make the dataset more managable and to release some memory. In a real-case situation, I would not have adopted such a conservative approach.
# 
# ### Remove columns with more than 60% null values
# These are:

# In[2]:


check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
check_null[check_null>0.6]


# ...for a total of 21 columns. We can remove these columns with `inplace=True` to overwrite the current dataframe in memory. Remove also a line with all null values...

# In[3]:


df.drop(check_null[check_null>0.5].index, axis=1, inplace=True) 
df.dropna(axis=0, thresh=30, inplace=True)


# ### Remove columns with little information
# Here are some columns we want to remove and why:
# 1. `index` is not needed because it's built-in the dataframe itself
# 2. `policy_code` is always `== 1`
# 3. `payment_plan` has only 10 `y` and 887372 `n`
# 4. `url` not needed, although it might be useful if it contains extra-data (e.g., payment history)
# 5. `id` and `member_id` are all unique, which is a bit misleading. I was expecting to find payment histories, but it seems that every record is a single customer.
# 6. `application_type` is 'INDIVIDUAL' for 99.94% of the records
# 7. `acc_now_delinq` is `0` for 99.5% of the records
# 8. `emp_title` not needed here, but it might be useful for the modelling (see below), 
# 9. `zip_code` not needed for this level of analysis,
# 10. `title` might be useful with NLP, but let's ignore it for now
# 
# Numbers above have been calculated by grouping by the metrics, counting the size of each group and sorting. For example:

# In[4]:


df.groupby('application_type').size().sort_values()


# In[5]:


delete_me = ['index', 'policy_code', 'pymnt_plan', 'url', 'id', 'member_id', 'application_type', 'acc_now_delinq','emp_title', 'zip_code','title']
df.drop(delete_me , axis=1, inplace=True) 


# We can now delete the columns above:

# ## Feature transformations

# The dataset has now 43 columns. We need to transform a few metrics which sound very important, but are formatted as strings. These transformations are performed with the __modelling__ in mind. Ultimatelly we want to produce a dataset almost ready to be fed to the model. Here is a summary of the operations performed:
# 1. Strip `months` from `term` and make it an integer
# 2. The Interest rate is a string. Remove `%` and make it a float
# 3. Extract numbers from `emp_length` and fill missing values with the median (see below). If `emp_length == 10+ years` then leave it as `10`
# 4. Transform `datetimes` to a Period 

# In[6]:


# strip months from 'term' and make it an int
df['term'] = df['term'].str.split(' ').str[1]

#interest rate is a string. Remove % and make it a float
df['int_rate'] = df['int_rate'].str.split('%').str[0]
df['int_rate'] = df.int_rate.astype(float)/100.

# extract numbers from emp_length and fill missing values with the median
df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)
df['emp_length'] = df['emp_length'].fillna(df.emp_length.median())

col_dates = df.dtypes[df.dtypes == 'datetime64[ns]'].index
for d in col_dates:
    df[d] = df[d].dt.to_period('M')


# ## Data exploration
# We now have the data in a more suitable form for data exploration. I could plot different combinations of metrics on 2-dimensional plots and look for interesting trends. Instead, I want to touch briefly two techniques that can allow us to have an overview of the dataset without too much coding involved.
# 
# ### Use interactive pivot tables with javascript
# We can explore the dataset with one single javascript wrapper using the library `pivottablejs` which allows us to do aggregations and plotting using javascipt libraries. On this computer, this library cannot handle 800k rows and 43 columns in a reasonable amount of time, so I decided to input a __random__ selection of 10% of the dataframe. This should be ok for proportions and averaged, but not for absolute counts. This is when aggregating in-database would speed things up.

# In[7]:


# pivot_ui(df.sample(frac=0.1))
# opens a new window


# A few things to notice:
# * A line plot of `issue_dt` vs. `grade` (counted as fraction of columns) reveals that the relative fraction of loan grade changes with time (especially after 2012-07). It would be interesting to understand if this change was due to business changes or to changes in customer behaviour. 
# * A stacked bar chart plot of `home_ownership` vs. `loan_status` (counted as fraction of columns) shows that a `loan status` of *Charged_off* is about 4% for customers who own, rent or with a mortgage. Even though the `loan_status` is 10% and 25% for customers with None or Other, the total counts for these categories are very small. 
# * A stacked bar chart plot of `grade` vs. `loan_status` (counted as fraction of columns) shows that, as expected, the *Charged_off* status becomes more and more relevant for higher interest rates (grades F and G)

# # Data Modelling

# __Let's build a model which predicts the status *charged_off*__. The fraction of this status in the whole dataset is low, only around 5%, but not as low as other status. 

# 
# The problem here is: __How well can we predict that a prospect customer will charge off at some point in the future?__ 

# ## More feature engeneering
# We can finally choose the metrics for the model remembering to check for missing values and transforming metrics in a way suitable for modelling. 
# 
# * Let's keep the `loan_amount`, but let's create a metric which indicates that the total amount committed by investors for that loan at that point in time (`funded_amnt_inv`) is less than what the borrower requested.

# In[8]:


df['amt_difference'] = 'eq'
df.loc[(df['funded_amnt'] - df['funded_amnt_inv']) > 0,'amt_difference'] = 'less'


# * The interest rate is an important metrics, but it changes with time, whereas the interest grade does not. So, we will consider the interest `grade` only, exluding the `sub_grade` to keep it simple.
# 
# * the metrics `delinq_2yrs` is very skewed towards zero (80% are zeros). Let's make it categorical: `no` when `delinq_2yrs == 0` and `yes` when  `delinq_2yrs > 0`
# 
# * Same as above for `inq_last_6mths`: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
# 
# * Same as above for `pub_rec`: Number of derogatory public records
# 
# * I thought about computing difference between the date of the earliest credit line and the issue date `df['tmp'] = df.earliest_cr_line - df.issue_d`, but I do not understand the metrics well, so I will skip this
# 
# * Let's compute the ratio of the number of open credit lines in the borrower's credit file divided by the total number of credit lines currently in the borrower's credit file

# In[9]:


# Make categorical

df['delinq_2yrs_cat'] = 'no'
df.loc[df['delinq_2yrs']> 0,'delinq_2yrs_cat'] = 'yes'

df['inq_last_6mths_cat'] = 'no'
df.loc[df['inq_last_6mths']> 0,'inq_last_6mths_cat'] = 'yes'

df['pub_rec_cat'] = 'no'
df.loc[df['pub_rec']> 0,'pub_rec_cat'] = 'yes'

# Create new metric
df['acc_ratio'] = df.open_acc / df.total_acc


# These are the features we want to model

# In[10]:


features = ['loan_amnt', 'term', 
            'installment', 'emp_length',
            'home_ownership', 'annual_inc','verification_status',
            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',  
            'loan_status'
           ]



# Given the business problem stated above, we want to distinguish between a customer who will *charge off* and a customer who will pay in full. I will not model the cohort of *Current* customers because these are still "in progress" and belong to the second scenario. 

# In[11]:


X_clean = df.loc[df.loan_status != 'Current', features]
X_clean.head()


# In[12]:


mask = (X_clean.loan_status == 'Charged Off')
X_clean['target'] = 0
X_clean.loc[mask,'target'] = 1
X_clean


# In[ ]:


#Prepare your xtest tuple here
xtest = {'loan_amnt': [50000], 'term': [36] ,  'installment': [500.36], 'emp_length': [3.0], 'home_ownership': "RENT", 'annual_inc': 45000.0, 'verification_status': "Not Verified",  'purpose': "small_business", 'dti': 8.72, 'delinq_2yrs_cat': "no", 'inq_last_6mths_cat': "yes", 'open_acc': 10.0, 'pub_rec': 0.0, 'pub_rec_cat': "no", 'acc_ratio': 0.67, 'initial_list_status': "f"}


# In[44]:


cat_features = ['term', 'home_ownership', 'verification_status', 'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'pub_rec_cat', 'initial_list_status']

# Drop any residual missing value (only 24)
X_clean.dropna(axis=0, how = 'any', inplace = True)

X = pd.get_dummies(X_clean[X_clean.columns[:-2]], columns=cat_features).astype(float)

#xtest = { 'loan_amnt': [50000], 'installment': [500.36] ,'emp_length': [3.0], 'annual_inc': [45000], 'dti': [8.72], 'open_acc': [10], 'pub_rec': [0.0], 'acc_ratio': [0.67], 'term_36': [1.0], 'term_60': [0.0], 'home_ownership_ANY': [0.0], 'home_ownership_MORTGAGE': [0.0],'home_ownership_NONE': [0.0], 'home_ownership_OTHER': [0.0], 'home_ownership_OWN':[0.0], 'home_ownership_RENT': [1.0], 'verification_status_Not Verified':[0.0] ,'verification_status_Source Verified': [1.0],'verification_status_Verified': [0.0],'purpose_car':[0.0] ,'purpose_credit_card':[0.0],'purpose_debt_consolidation':[0.0] ,'purpose_educational': [0.0],'purpose_home_improvement': [0.0],'purpose_house':[0.0] ,'purpose_major_purchase':[0.0] ,'purpose_medical': [0.0],'purpose_moving': [0.0],'purpose_other': [0.0],'purpose_renewable_energy':[0.0] ,'purpose_small_business': [1.0],'purpose_vacation': [0.0], 'purpose_wedding': [0.0], 'delinq_2yrs_cat_no': [1.0], 'delinq_2yrs_cat_yes': [0.0] , 'inq_last_6mths_cat_no': [0.0], 'inq_last_6mths_cat_yes': [1.0], 'pub_rec_cat_no':[1.0] , 'pub_rec_cat_yes': [0.0], 'initial_list_status_f':[1.0] , 'initial_list_status_w':[0.0]}
df2 = pd.DataFrame(data=xtest)[:]
X.append(df2, ignore_index=True)
y = X_clean['target']
#datafile = 'saved_dataframe'
#df.to_pickle(datafile)  
#df = pd.read_pickle(datafile)


# ## The models
# 
# Let's start modelling by importing a few modules. Features are all on different scale, so it is wise to rescale all features in the range -1, +1

# In[45]:


import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

X_scaled = preprocessing.scale(X)

xtest = X_scaled[-1, :]
np.delete(X_scaled, -1, 0)

#print('   ')
print(X_scaled.shape)


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=0)
index_split = int(len(X)/2)
X_train, y_train = SMOTE().fit_sample(X_scaled[0:index_split, :], y[0:index_split])
X_test, y_test = X_scaled[index_split:], y[index_split:]


# Write a function that :
# 1. Takes train and test set under different assumptions
# 2. Runs a set of models. 3 in this case: Gradient Boosting, Logistic Regression and Random Forest
# 3. Makes prediction using the test set
# 4. Builds-up a table with evaluation metrics
# 5. Plots a roc curve of the estimators

# In[47]:



    #cols = ['model','matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score','f1_score']

    #models_report = pd.DataFrame(columns = cols)
conf_matrix = dict()
clf = LogisticRegression()
clf.fit(X_train, y_train)
    #clf.fit(X_train, y_train)
    
#y_pred = clf.predict()
y_score = clf.predict_proba(X_test)[:,1]

    


# In[48]:


# save the model to disk
filename = 'modelPickle.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[49]:


xtest = xtest.transpose()
xtest.shape


# In[50]:


#load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

y_pred = loaded_model.predict_proba(xtest)[:,1]
y_pred


# ### Model with unbalanced classes
# If we do not modify the class ratios our model has very poor predictive power. The area ander the curve (AUC) is about 0.6, suggesting that we perform better than random. However, the recall is zero: we cannot predict the target variable at all. This might be either because there is something wrong with the metrics or because the classes are too unbalanced. 

# ### Model with synthetically balanced classes
# 
# We can artificially balance the classes using the algorithm SMOTE ( Synthetic Minority Over-sampling Technique). This uses a K-nearest neighbour approach to create feature vectors which resemble those of the target variable. The minority class is oversampled. With this trick, the performance of the model improves considerably.
# 
# We now have a recall of 70% using Logistic Regression. We get right 7 out of 10 customers who will "charge off". On the other hand we have a precision of 20%. 
