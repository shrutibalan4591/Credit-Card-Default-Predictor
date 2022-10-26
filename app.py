
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#import bz2file as bz2

"""
# Method for encoding gender value
def gender_encode(gender):
  if gender==1:
    return([1,0])
  else:
    return([0,1])

# Method for encoding education value
def education_encode(education):
  if education==1:
    return([0,1,0,0])
  elif education==2:
    return([0,0,1,0])
  elif education==3:
    return([1,0,0,0])
  elif education==4:
    return([0,0,0,1])

# Method for encoding education value
def marital_encode(marital_status):
  if marital_status==1:
    return([0,1,0])
  elif marital_status==2:
    return([1,0,0])
  elif marital_status==3:
    return([0,0,1])


# Method for encoding repayment status values
def rs_encode(rs):
  if rs==-2:
    return([1,0,0,0,0,0,0,0,0,0,0,0])
  elif rs==-1:
    return([0,1,0,0,0,0,0,0,0,0,0,0])
  elif rs==0:
    return([0,0,1,0,0,0,0,0,0,0,0,0])
  elif rs==1:
    return([0,0,0,1,0,0,0,0,0,0,0,0])
  elif rs==2:
    return([0,0,0,0,1,0,0,0,0,0,0,0])
  elif rs==3:
    return([0,0,0,0,0,1,0,0,0,0,0,0])
  elif rs==4:
    return([0,0,0,0,0,0,1,0,0,0,0,0])
  elif rs==5:
    return([0,0,0,0,0,0,0,1,0,0,0,0])
  elif rs==6:
    return([0,0,0,0,0,0,0,0,1,0,0,0])
  elif rs==7:
    return([0,0,0,0,0,0,0,0,0,1,0,0])
  elif rs==8:
    return([0,0,0,0,0,0,0,0,0,0,1,0])
  elif rs==9:
    return([0,0,0,0,0,0,0,0,0,0,0,1])      
"""

app = Flask(__name__)
def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = pickle.load(data)
  return data

model = decompress_pickle('ccdp.pbz2')
#model = pickle.load(open('ccdp.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('myindex.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        gender = gender_encode(int(request.form['gender']))
        education = education_encode(int(request.form['education']))
        marital_status = marital_encode(int(request.form['marriage']))
        age = [int(request.form['age'])]
        bal_limit = [int(request.form['limit_bal'])]
        rs_6 = rs_encode(int(request.form['april_rs']))
        rs_5 = rs_encode(int(request.form['may_rs']))
        rs_4 = rs_encode(int(request.form['june_rs']))
        rs_3 = rs_encode(int(request.form['july_rs']))
        rs_2 = rs_encode(int(request.form['august_rs']))
        rs_1 = rs_encode(int(request.form['september_rs']))
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

    bill_amt_avg = (bill_6 + bill_5 + bill_4 + bill_3 + bill_2 + bill_1)/6
    #bill_pay_value = ((pay_1-bill_2) + (pay_2-bill_3) + (pay_3-bill_4) + (pay_4-bill_5) + (pay_5-bill_6))/5
    #bill_pay_pn = int(np.where(bill_pay_value<=0, 0, 1))

    features = rs_1 + rs_2 + pay_1 + bill_1
    features = features + limit + age + pay_2 + bill_2
    features = features + pay_3 + bill_3 + bill_4 + pay_4
    features = features + pay_6 + bill_5 + bill_6 + bill_amt_avg
 
    features_arr = [np.array(features)]

    prediction = model.predict(features_arr)

    result = ""
    if prediction == 1:
      result = "This customer IS LIKELY TO DEFAULT next month."
    else:
      result = "This customer IS NOT LIKELY TO DEFAULT next month."


    return render_template('myindex.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

