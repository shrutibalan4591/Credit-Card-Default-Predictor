import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import bz2file as bz2

app = Flask(__name__)
def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = pickle.load(data)
  return data

model = decompress_pickle('ccdp.pbz2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        #gender = gender_encode(int(request.form['gender']))
        #education = education_encode(int(request.form['education']))
        #marital_status = marital_encode(int(request.form['marriage']))
        age = [int(request.form['age'])]
        bal_limit = [int(request.form['limit_bal'])]
        rs_6 = [int(request.form['april_rs'])]
        rs_5 = [int(request.form['may_rs'])]
        rs_4 = [int(request.form['june_rs'])]
        rs_3 = [int(request.form['july_rs'])]
        rs_2 = [int(request.form['august_rs'])]
        rs_1 = [int(request.form['september_rs'])]
        bill_6 = [int(request.form['bill_amt6'])]
        bill_5 = [int(request.form['bill_amt5'])]
        bill_4 = [int(request.form['bill_amt4'])]
        bill_3 = [int(request.form['bill_amt3'])]
        bill_2 = [int(request.form['bill_amt2'])]
        bill_1 = [int(request.form['bill_amt1'])]
        pay_6 = [int(request.form['pay_amt6'])]
        pay_5 = [int(request.form['pay_amt5'])]
        pay_4 = [int(request.form['pay_amt4'])]
        pay_3 = [int(request.form['pay_amt3'])]
        pay_2 = [int(request.form['pay_amt2'])]
        pay_1 = [int(request.form['pay_amt1'])]

    bill_amt_avg = [round(np.mean([bill_6, bill_5, bill_4, bill_3, bill_2, bill_1]), 2)]
    features = rs_1 + rs_2 + pay_1 + bill_1
    features = features + bal_limit + age + pay_2 + bill_2
    features = features + pay_3 + bill_3 + bill_4 + pay_4
    features = features + pay_6 + bill_5 + bill_6 + bill_amt_avg
    
    features_arr = [np.array(features)]

    prediction = model.predict(features_arr)

    #result = ""
    #if prediction == 1:
    #  result = "This customer IS LIKELY TO DEFAULT next month."
    #else:
    #  result = "This customer IS NOT LIKELY TO DEFAULT next month."


    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
