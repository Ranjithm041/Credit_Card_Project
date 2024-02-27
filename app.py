
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import phonenumbers
from Test_file import number
from phonenumbers import timezone,geocoder
ch_number = phonenumbers.parse(number,"CH")

filename = 'CreditCard-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        time = int(request.form['time'])
        cardtype = int(request.form['cardtype'])
        bankname=int(request.form['bankname'])
        cardno = request.form['cardno']
        c1=int(cardno[0])
        c2=int(cardno[1])
        c3=int(cardno[2])
        c4=int(cardno[3])
        c5=int(cardno[5])
        c6=int(cardno[6])
        c7=int(cardno[7])
        c8=int(cardno[8])
        c9=int(cardno[10])
        c10=int(cardno[11])
        c11=int(cardno[12])
        c12=int(cardno[13])
        c13=int(cardno[15])
        c14=int(cardno[16])
        c15=int(cardno[17])
        c16=int(cardno[18])
        transid = int(request.form['transid'])
        cvvno =request.form['cvvno']
        cvv1=int(cvvno[0])
        cvv2=int(cvvno[1])
        cvv3=int(cvvno[2])
        month = request.form['month']
        m1=int(month[0])
        m2=int(month[1])
        year = request.form['year']
        y1=int(year[0])
        y2=int(year[1])
        amount = float(request.form['amount'])
        enc = OneHotEncoder()
        X = enc.fit_transform([[0, 0, bankname,cardtype],[c1,c2,c3,c4],[c5,c6,c7,c8],[c9,c10,c11,c12],[c13,c14,c15,c16],[transid,cvv1,cvv2,cvv3],[m1,m2,y1,y2]]).toarray()
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(X)
        data = np.array([[time, X_pca[0][0], X_pca[0][1], X_pca[0][2], X_pca[0][3], X_pca[1][0], X_pca[1][1], X_pca[1][2], X_pca[1][3], X_pca[2][0], X_pca[2][1], X_pca[2][2], X_pca[2][3], X_pca[3][0], X_pca[3][1], X_pca[3][2], X_pca[3][3], X_pca[4][0], X_pca[4][1], X_pca[4][2], X_pca[4][3], X_pca[5][0], X_pca[5][1], X_pca[5][2], X_pca[5][3], X_pca[6][0], X_pca[6][1], X_pca[6][2], X_pca[6][3], amount]])
        print(data)
        my_prediction = classifier.predict(data)
        if cardtype>=4:
            my_prediction=1
        if bankname>=3:
            my_prediction=1
        if c1==0:
            my_prediction=1
        if amount<=100:
            my_prediction=1
        print(my_prediction)
        print(geocoder.description_for_number(ch_number, "en"))
        return render_template('result.html', prediction=my_prediction)
        #return render_template('result.html')
if __name__ == '__main__':
	app.run(debug=True)