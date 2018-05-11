from flask import Flask, render_template, json, request, jsonify
from sklearn.externals import joblib
import pandas as pd
import sqlite3
import pandas as pd 
import numpy as np 
import os 
from flask import send_from_directory 

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

conn = sqlite3.connect('./Database/database.sqlite') # This might take a while to run...
to_parse = ['issue_d' , 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
df = pd.read_sql_query('select * from loan', con=conn, parse_dates = to_parse)
check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
check_null[check_null>0.6]
df.drop(check_null[check_null>0.5].index, axis=1, inplace=True) 
df.dropna(axis=0, thresh=30, inplace=True)
df.groupby('application_type').size().sort_values()
delete_me = ['index', 'policy_code', 'pymnt_plan', 'url', 'id', 'member_id', 'application_type', 'acc_now_delinq','emp_title', 'zip_code','title']
df.drop(delete_me , axis=1, inplace=True)
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
df['amt_difference'] = 'eq'
df.loc[(df['funded_amnt'] - df['funded_amnt_inv']) > 0,'amt_difference'] = 'less'

df['delinq_2yrs_cat'] = 'no'
df.loc[df['delinq_2yrs']> 0,'delinq_2yrs_cat'] = 'yes'

df['inq_last_6mths_cat'] = 'no'
df.loc[df['inq_last_6mths']> 0,'inq_last_6mths_cat'] = 'yes'

df['pub_rec_cat'] = 'no'
df.loc[df['pub_rec']> 0,'pub_rec_cat'] = 'yes'

# Create new metric
df['acc_ratio'] = df.open_acc / df.total_acc

# These are the features we want to model

# In[11]:


features = ['loan_amnt', 'amt_difference', 'term', 
            'installment', 'grade','emp_length',
            'home_ownership', 'annual_inc','verification_status',
            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',  
            'loan_status'
           ]


#Prepare your xtest tuple here
xtest = {'loan_amnt': [50000], 'term': [36] ,  'installment': [500.36], 'emp_length': [3.0], 'home_ownership': "RENT", 'annual_inc': 45000.0, 'verification_status': "Not Verified",  'purpose': "small_business", 'dti': 8.72, 'delinq_2yrs_cat': "no", 'inq_last_6mths_cat': "yes", 'open_acc': 10.0, 'pub_rec': 0.0, 'pub_rec_cat': "no", 'acc_ratio': 0.67, 'initial_list_status': "f"}


X_clean = df.loc[df.loan_status != 'Current', features]

mask = (X_clean.loan_status == 'Charged Off')
X_clean['target'] = 0
X_clean.loc[mask,'target'] = 1


cat_features = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status', 'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'pub_rec_cat', 'initial_list_status']

# Drop any residual missing value (only 24)
X_clean.dropna(axis=0, how = 'any', inplace = True)

X = pd.get_dummies(X_clean[X_clean.columns[:-2]], columns=cat_features).astype(float)
test_df = pd.DataFrame(data=xtest)[:]
X.append(test_df, ignore_index=True)
y = X_clean['target']



X_scaled = preprocessing.scale(X)
xtest = X_scaled[-1, :]
np.delete(X_scaled, -1, 0)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.4, random_state=0)
index_split = int(len(X)/2)
X_train, y_train = SMOTE().fit_sample(X_scaled[0:index_split, :], y[0:index_split])
X_test, y_test = X_scaled[index_split:], y[index_split:]

clf = LogisticRegression()
clf.fit(X_train, y_train)
   
    
#y_pred = clf.predict()
y_score = clf.predict_proba(X_test)[:,1]



# save the model to disk
filename = 'modelPickle.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[26]:
xtest = xtest.transpose()
xtest.shape


#load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_score = loaded_model.predict_proba(xtest)[:,1]
print ("yes loaded")
print (y_score)

app = Flask(__name__)

@app.route("/")
def main():
	 return render_template('index.html')

@app.route('/startScreen')
def startScreen():
    return render_template('index.html')

@app.route("/getPrediction")
def getPrediction():
	
	if _name and _loan_amount and _term:
	  return render_template('success.html')
	else:
	  return render_template('failure.html')

@app.route('/showSignUp')
def showSignUp():
    return render_template('signup.html')

@app.route("/signUp", methods=['POST'])
def signUp():
	_name = request.form
	_loan_amount = request.form['inputAmount']
	_term = request.form['term']
	if _name and _loan_amount and _term:
	  return render_template('success.html')
	else:
	  return render_template('failure.html')

 
@app.route('/favicon.ico') 
def favicon(): 
	return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
 
@app.route("/members/<string:name>/")
def getMember(name):
    return name
 
if __name__ == "__main__":
    app.run()
