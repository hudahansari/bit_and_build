import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(a) for a in request.form.values()]
    x = np.array(float_features)
    y=np.array([1.0,2.0,3.0,4.0,5.0,6.0])
    if x[0]>40:
        y[0]=1
    else:
        y[0]=0
    
    if x[1]>40:
        y[1]=1
    else:
        y[1]=0

    if x[2]>40:
        y[2]=1
    else:
        y[2]=0
    
    y[3]=x[0]+x[1]+x[2]
    y[4]=y[3]/3
    if y[4]>40:
        y[5]=1
    else:
        y[5]=0

    x[0]=(x[0]-18)/(100-18)
    x[1]=(x[1]-24)/(100-24)
    x[2]=(x[2]-15)/(100-15)
    # y[3]=y[3]
    y[3]=(y[3]-69.0)/(300.0-69.0)
    # y[3]=round(y[3],5)
    y[4]=(y[4]-23)/(100-23)
    # return '{}'.format(y[3].dtype)



    features=np.append(x,y)
    # print(features)
    # mm = MinMaxScaler()
    # # model = scaler.fit(features)
    # # scaled_data = model.transform(features)
    # scaled_data=mm.fit_transform(features)
    features=features.reshape(1,-1)
    # return '{}'.format(features)
    prediction = model.predict(features)
    # return '{}'.format(prediction[0])
    # prediction_text=format(prediction)
    # return '{}'.format(prediction_text)
    # if prediction_text == '0' :
    #     prediction = 'O'
    # elif prediction_text == '1' :
    #     prediction = 'A'
    # elif prediction_text == '2' :
    #     prediction = 'B'
    # elif prediction_text == '3' :
    #     prediction = 'C'
    # elif prediction_text == '4' :
    #     prediction = 'D'
    # else :
    #     prediction = 'E'

    # return render_template("index.html",prediction)

    return render_template("index.html", prediction_text = format(prediction[0]))

if __name__ == "__main__":
    flask_app.run(debug=True)