
# Importing the necessary Libraries

from Batch_process.Batch_process import batch_data_prep
import xgboost
import pickle
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import os
import json
from flask import Flask, render_template, request,jsonify
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import smtplib
from email.message import EmailMessage
from config_reader import read_config
#https://stackoverflow.com/questions/49921721/runtimeerror-main-thread-is-not-in-main-loop-with-matplotlib-and-flask
#to avoid the error ---The Exception message is:  main thread is not in main loop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import csv

# import request
app = Flask(__name__)

# initialising the flask app with the name 'app'

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")



@app.route('/predict_batch', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def predict_batch():
    if request.method == 'POST':
        try:
            df = pd.read_csv(request.files["file"])
            email_id = request.form["email"]
            print(email_id)
            if 'Unnamed: 0' in df.columns:
                test_df = df.drop(['Unnamed: 0'], axis=1)
            else:
                test_df = df
            #print(test_df.head(2))
            prepped_df_object = batch_data_prep(test_df, 20)
            categorical_feature_list = prepped_df_object.categorical_feature(test_df, 20)
            prepped_df_object.data_visualization(test_df, categorical_feature_list, cols=2, width=20, height=45, hspace=0.8, wspace=0.8)
            encoded_df = prepped_df_object.encoder(test_df)
            encoded_df.head(2)
            prepped_df_object.data_visualization_groupby_target(encoded_df, categorical_feature_list, cols=2, width=20, height=45,
                                                   hspace=0.8, wspace=0.8)
            list_of_files = os.listdir('static')
            list_of_jpg_files = list_of_files[1:]
            X = encoded_df.drop('wage_class', axis=1)
            filename = 'xgboost_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
            # predictions using the loaded model file
            prediction = loaded_model.predict(X)
            X['Predictions'] = pd.Series(prediction)
            X['Predictions'] = X['Predictions'].map({1: '>50k', 0: '<=50k'})
            X.to_csv('result.csv')
            sendEmail(email_id)
            return render_template('results.html', prediction=f'We sent you an email with all the predictions in which the first five predictions are: {str(list(prediction[0:5]))}',user_images = list_of_jpg_files)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


def single_item_predict(age, workclass, fnlwgt, education, education_num, marital_status,occupation,relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country):
    filename = 'xgboost_model.pickle'
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    # predictions using the loaded model file
    prediction = loaded_model.predict(pd.DataFrame({'age':age,'workclass': workclass,'fnlwgt':fnlwgt,'education': education,'education_num':education_num,'marital_status': marital_status,'occupation':occupation, 'relationship':relationship,'race': race,'sex': sex, 'capital_gain':capital_gain, 'capital_loss': capital_loss,'hours_per_week': hours_per_week,'native_country': native_country}, index = [0]))
    return prediction[0]


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            age=float(request.form['age'])
            is_workclass = request.form['workclass']
            fnlwgt = float(request.form['fnlwgt'])
            is_education = request.form['education']
            education_num = float(request.form['education_num'])
            is_marital_status = request.form['marital_status']
            is_occupation = request.form['occupation']
            is_relationship = request.form['relationship']
            is_race = request.form['race']
            is_sex = request.form['sex']
            capital_gain = float(request.form['capital_gain'])
            capital_loss = float(request.form['capital_loss'])
            hours_per_week = float(request.form['hours_per_week'])
            is_native_country = request.form['native_country']

            if(is_sex=='Male'):
                sex=1
            else:
                sex=0

            dict_workclass = {'Self-emp-not-inc': 0, 'Private': 1, 'State-gov': 2, 'Federal-gov': 3, 'Local-gov': 4, '?': 5, 'Self-emp-inc': 6, 'Without-pay': 7,
                              'Never-worked': 8}
            for i in dict_workclass:
                if is_workclass == i:
                    workclass = dict_workclass[i]
                else:
                    continue

            dict_education = {'Bachelors': 0,'HS-grad': 1,'11th': 2,'Masters': 3,'9th': 4,'Some-college': 5,
                              'Assoc-acdm': 6,'Assoc-voc': 7,'7th-8th': 8,'Doctorate': 9,'Prof-school': 10,
                              '5th-6th': 11, '10th': 12,'1st-4th': 13,'Preschool': 14,'12th': 15}
            for i in dict_education:
                if is_education == i:
                    education = dict_education[i]
                else:
                    continue

            dict_marital_status = {'Married-civ-spouse': 0,'Divorced': 1,'Married-spouse-absent': 2,
                                   'Never-married': 3,'Separated': 4,'Married-AF-spouse': 5,'Widowed': 6}

            for i in dict_marital_status:
                if is_marital_status == i:
                    marital_status = dict_marital_status[i]
                else:
                    continue

            dict_occupation = {'Exec-managerial': 0,'Handlers-cleaners': 1,'Prof-specialty': 2,
                              'Other-service': 3,'Adm-clerical': 4,'Sales': 5,'Craft-repair': 6,
                              'Transport-moving': 7,'Farming-fishing': 8,'Machine-op-inspct': 9,
                              'Tech-support': 10,'?': 11,'Protective-serv': 12,'Armed-Forces': 13,
                              'Priv-house-serv': 14}

            for i in dict_occupation:
                if is_occupation == i:
                    occupation = dict_occupation[i]
                else:
                    continue

            dict_relationship = {'Husband': 0,'Not-in-family': 1,'Wife': 2,'Own-child': 3,'Unmarried': 4, 'Other-relative': 5}

            for i in dict_relationship:
                if is_relationship == i:
                    relationship = dict_relationship[i]
                else:
                    continue

            dict_race = {'White': 0,'Black': 1,'Asian-Pac-Islander': 2,'Amer-Indian-Eskimo': 3,'Other': 4}

            for i in dict_race:
                if is_race == i:
                    race = dict_race[i]
                else:
                    continue

            dict_native_country = {'United-States': 0,'Cuba': 1,'Jamaica': 2,'India': 3,'?': 4,'Mexico': 5,
                                   'South': 6,'Puerto-Rico': 7,'Honduras': 8,'England': 9,'Canada': 10,
                                   'Germany': 11,'Iran': 12,'Philippines': 13,'Italy': 14,'Poland': 15,
                                   'Columbia': 16,'Cambodia': 17,'Thailand': 18,'Ecuador': 19,'Laos': 20,
                                   'Taiwan': 21,'Haiti': 22,'Portugal': 23,'Dominican-Republic': 24,
                                   'El-Salvador': 25,'France': 26,'Guatemala': 27,'China': 28,'Japan': 29,
                                   'Yugoslavia': 30,'Peru': 31,'Outlying-US(Guam-USVI-etc)': 32,
                                   'Scotland': 33,'Trinadad&Tobago': 34,'Greece': 35,'Nicaragua': 36,
                                   'Vietnam': 37,'Hong': 38,'Ireland': 39,'Hungary': 40,'Holand-Netherlands': 41}
            for i in dict_native_country:
                if is_native_country == i:
                    native_country = dict_native_country[i]
                else:
                    continue

            prediction = single_item_predict(age, workclass, fnlwgt, education, education_num, marital_status,occupation,relationship, race, sex, capital_gain, capital_loss, hours_per_week,native_country)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            if prediction == 0:
                return render_template('results.html',prediction='<=50k')
            else:
                return render_template('results.html', prediction='>50k')
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')

# testing via postman

@app.route('/from_postman',methods=['POST']) # route to show the predictions in a web UI
@cross_origin()
def from_postman():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            gre_score=float(request.json['gre_score'])
            toefl_score = float(request.json['toefl_score'])
            university_rating = float(request.json['university_rating'])
            sop = float(request.json['sop'])
            lor = float(request.json['lor'])
            cgpa = float(request.json['cgpa'])
            is_research = request.json['research']
            if(is_research=='yes'):
                research=1
            else:
                research=0
            prediction = predict(gre_score, toefl_score, university_rating, sop, lor, cgpa, research)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return jsonify({'Prediction':round(100*prediction[0])})
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


# Helper functions
def emptyTargetFiles(files_list):
    if len(files_list) > 0:
        for i in files_list:
            if 'pdf' in i or 'csv' in i:
                os.remove(i)
            else:
                continue


def sendEmail(email_id):
    #if request.form['EmailId'] is not None:
    msg = EmailMessage()
    msg['To'] = email_id
    print(msg['To'] )
    readConfig = read_config()
    msg['From'] = readConfig['SENDER_EMAIL']
    pswd = readConfig['PASSWORD']
    msg.set_content(readConfig['EMAIL_BODY'])
    msg['Subject'] = readConfig['EMAIL_SUBJECT']
    list_of_files = os.listdir()
    if 'result.csv' in list_of_files:
        files = ['result.csv']
        for file in files:
            with open(file, 'rb') as f:
                file_data = f.read()
                file_name = f.name
            msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(msg['From'], pswd)
            smtp.send_message(msg)
    emptyTargetFiles(list_of_files)
        #return {
        #    "fulfillmentText": speech,
        #    "displayText": speech
        #}


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app