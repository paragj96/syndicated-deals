from flask import Flask, render_template, json, request, jsonify
from sklearn.ensemble import BaggingClassifier
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle
import patsy
import pandas as pd
import os 
from collections import OrderedDict
from flask import send_from_directory 
from sklearn.preprocessing import LabelEncoder
import requests
from datetime import datetime
from django.http import JsonResponse

df = pickle.load(open("cleaned_data", 'rb'))
df = df.drop(['emp_title','title','addr_state','issue_d'], axis=1)
df['term'] = df['term'].str.split(' ').str[1]

'''filename1 = 'X_after_smote'
X1 = pickle.load(open(filename1, 'rb'))
filename2 = 'Y_after_smote'
Y1 = pickle.load(open(filename2, 'rb'))
X_train,X_test, Y_train, Y_test = train_test_split(X1,Y1,test_size=0.3)
'''
filename3 = 'bagged_model'
bagging = pickle.load(open(filename3, 'rb'))



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
        ts1 = datetime.now() 
        df = pickle.load(open("cleaned_data", 'rb'))
        df = df.drop(['emp_title','title','addr_state','issue_d'], axis=1)
        df['term'] = df['term'].str.split(' ').str[1]
        _listing_id = request.form['listingId']
        _reserveRoi = request.form['reserveRoi']
        _term = request.form['term']
        _installment = request.form['installment']
        _emp_length = request.form['emp_length']
        _property_ownership = request.form['property_ownership']
        _annual_revenue = request.form['annual_revenue']
        _purpose = request.form['purpose']
        _dti = request.form['dti']
        _delinq_2yrs = request.form['delinq_2yrs']
        _inq_last_6mths = request.form['inq_last_6mths']
        _open_acc = request.form['open_acc']
        _total_acc = request.form['total_acc']
        _pub_rec = request.form['pub_rec']
        _grade = request.form['grade']
        _loan_amount = request.form['inputAmount']
        _int_rate = request.form['int_rate']
        _revol_bal = request.form['revol_bal']
        _revol_util = request.form['revol_util']
        _collections_12_mths_ex_med = request.form['collections_12_mths_ex_med']
        _acc_now_delinq = request.form['acc_now_delinq']
        _loan_id = request.form['lin']

        _int_rate = float(_int_rate)
	#_listing_id = int(_listing_id)
        _reserveRoi = float(_reserveRoi)
	
        _inq_last_6mths = int(_inq_last_6mths)
        _term = int(_term)
        _loan_amount = int(_loan_amount)
	
        _collections_12_mths_ex_med = int(_collections_12_mths_ex_med)
        _acc_now_delinq = int(_acc_now_delinq)
        _total_acc = int(_total_acc)


	#Prepare your xtest tuple here
        xdict = OrderedDict()
        xdict['loan_amnt']= [_loan_amount]
        xdict['term']= [_term]
        xdict['int_rate']= [_int_rate] 
        xdict['installment']= [_installment]
        xdict['grade']= [_grade]
        xdict['home_ownership']=[_property_ownership]
        xdict['inq_last_6mths']= [_inq_last_6mths]
	
        xtest = pd.DataFrame(xdict)
	
	
        df_local = df

        def GradeInt(x):
            if x == "A":
              return 1
            elif x == "B":
              return 2
            elif x == "C":
              return 3
            elif x == "D":
              return 4
            elif x == "E":
              return 5
            else:
              return 6

        xtest['GradeInt'] = xtest['grade'].map(GradeInt)
        xtest['Late_Loan'] = 0
        xtest = xtest[['GradeInt','inq_last_6mths','term', 'home_ownership', 'int_rate', 'Late_Loan']]
        le=LabelEncoder()
	# Iterating over all the common columns in train and test
        for col in xtest.columns.values:
	       # Encoding only categorical variables  ###Use whole data to form an exhaustive list of levels
            if xtest[col].dtypes=='object':
               data=df_local[col].append(xtest[col])
               le.fit(data.values)
               df_local[col]=le.transform(df_local[col])
               xtest[col]=le.transform(xtest[col])
        xtest = xtest.drop(['Late_Loan'], axis=1)
  
        print ("yes loaded")
        _willDefault = bagging.predict(xtest)
        print (bagging.predict(xtest))
        dataProperty = OrderedDict()
        dataProperty["$class"] = "org.marnet.loan.auction.LoanListing"
        dataProperty["listingId"]=_listing_id
        dataProperty["reserveRoi"]= 50
        dataProperty["state"]= "FOR_SALE"
        dataProperty["term"] = "36 months"
        dataProperty["installment"] = 150
        dataProperty["grade"] = str(_willDefault[0])
        dataProperty["emp_length"] = "10"
        dataProperty["property_ownership"] = "OWN"
        dataProperty["annual_revenue"] = 11200
        dataProperty["verification_status"] = "Verified"
        dataProperty["purpose"] = "car"
        dataProperty["dti"] = 21
        dataProperty["deling_2yrs"] = 0
        dataProperty["inq_last_6mths"] = 0
        dataProperty["open_acc"] = 10
        dataProperty["totol_acc"] = 28
        dataProperty["pub_rec"] = 1
        dataProperty["initial_list_status"] = "w"
        dataProperty["loan_status"] = "ok"
        dataProperty["loan"] = "org.marnet.loan.auction.Loan#" + _loan_id
                
        dataProperty2 = {"$class": "org.marnet.loan.auction.LoanListing",
                    "listingId": _listing_id,
                    "reserveRoi": 50,
                    "state": "FOR_SALE",
                    "term": "36 months",
                    "installment": 150,
                    "grade": str(_willDefault[0]),
                    "emp_length": "10",
                    "property_ownership": "OWN",
                    "annual_revenue": 11200,
                    "verification_status": "Verified",
                    "purpose": "car",
                    "dti": 21,
                    "deling_2yrs": 0,
                    "inq_last_6mths": 0,
                    "open_acc": 10,
                    "totol_acc": 28,
                    "pub_rec": 1,
                    "initial_list_status": "w",
                    "loan_status": "ok",
                    "loan": "org.marnet.loan.auction.Loan#" + _loan_id		
                }

        print(dataProperty)	
      	#add your block chain server ip here
        r = requests.post("http://13.126.39.191:3000/api/org.marnet.loan.auction.LoanListing", json = dataProperty2 )
        ts2 = datetime.now()
	#print(ts2-ts1)
        print(r.status_code, r.reason)
        if _listing_id and _loan_amount and _term:
          return render_template('success.html')
        else:
          return render_template('failure.html')

 
@app.route('/favicon.ico') 
def favicon(): 
	return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
 

 
if __name__ == "__main__":
    app.run()
