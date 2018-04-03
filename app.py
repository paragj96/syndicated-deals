from flask import Flask, render_template, json, request, jsonify
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)

@app.route("/")
def main():
	return render_template('index.html')

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

 
@app.route("/hello")
def hello():
    return "Hello World!"
 
@app.route("/members")
def members():
    return "Members"
 
@app.route("/members/<string:name>/")
def getMember(name):
    return name
 
if __name__ == "__main__":
    app.run()
