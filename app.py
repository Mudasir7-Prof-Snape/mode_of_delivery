from flask import Flask, render_template, request,url_for
import pandas as pd
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from tensorflow.keras.models import load_model

app=Flask(__name__)
var=load_model("model.h5")
scaler = pickle.load(open("scalar.pkl",'rb'))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    age=int(request.form.get('age'))
    obs = int(request.form.get('obs'))
    bmi = int(request.form.get('bmi'))
    cons = int(request.form.get('pos'))
    pos = int(request.form.get('cons'))
    eff = int(request.form.get('eff'))
    dial = int(request.form.get('dial'))
    stat = int(request.form.get('stat'))
    bis = int(request.form.get('bis'))
    cervl = int(request.form.get('cervl'))
    cervw = int(request.form.get('cervw'))
    induc = int(request.form.get('induc'))
    print(age,obs, bmi, cons, pos, eff, dial, stat, bis, cervl, cervw, induc)
    uinp=[[age,obs,bmi,cons,pos,eff,dial,stat,bis,cervl,cervw,induc]]
    df=pd.DataFrame(uinp,columns=['Age','Ob_Score','BMI','Consistency_Score','Position_Score','Effacement_Score','Dialation_Score','Station_Score','Total_Bishop_Score','Cerv_Len_cms','Cerv_Wid_cms','Induction'])
    sdf=scaler.transform(df)
    prediction=var.predict(sdf)[0]
    output="Normal Delivery"
    if prediction>=0.5:
        output="Cesarean"
    #print(output)
    #print(str(prediction))
    return output

if __name__=="__main__":
    app.run(debug=True,port=5001)
