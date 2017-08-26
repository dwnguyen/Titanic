#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:16:15 2017

@author: davidnguyen

Capable of achieving a .79904 accuracy on the test set of the Titanic Kaggle competition
"""

import tensorflow as tf
import math
import tflearn as tfl
import numpy as np
import pandas as pd

'''
#Get SigOP experiment for optimization purposes
from sigopt import Connection
conn = Connection(client_token= INSERT CLIENT TOKEN HERE )
experiment = conn.experiments(22981).fetch()
'''
tf.reset_default_graph()

# Load Data
df_data = pd.read_csv('train.csv', sep=',', usecols = [0, 2, 3, 4, 5, 6, 7, 8 ,9 ,10,11])
df_test = pd.read_csv('test.csv', sep=',', header = 0)

#Loads labels (Survived data)
_, labelsT = tfl.data_utils.load_csv('train.csv', target_column=1, columns_to_ignore = range(1), has_header = True, 
                        categorical_labels=True, n_classes=2 )
labels = labelsT[0:591][:]
labelsCV = labelsT[591:][:]


#Fills any remaining unknown ages and replaces ages with bins of age ranges
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

#Fills in unknown cabin values
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

#Replaces fare floats with quartile ranges
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

#Creates new column for number of family members  
def simplify_fam(df):
    df['numFam'] = df['SibSp'] +df['Parch']
    return df

#Uses name prefixes to guess unknown ages
def format_name(df):
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    for i in range(df['Pclass'].size):
        if math.isnan(df.iloc[i]['Age']):
            if df.iloc[i]['NamePrefix'] == 'Miss.':
                df = df.set_value(i, 'Age', 21.784417475728155)
            elif df.iloc[i]['NamePrefix'] == 'Mr.':
                df = df.set_value(i, 'Age', 32.282918149466191)
            elif df.iloc[i]['NamePrefix'] == 'Mrs.':
                df = df.set_value(i, 'Age', 37.231707317073173)
            elif df.iloc[i]['NamePrefix'] == 'Master.':
                df = df.set_value(i, 'Age', 5.3669230769230776)
            elif df.iloc[i]['NamePrefix'] == 'Rev.':
                df = df.set_value(i, 'Age', 41.25)
            elif df.iloc[i]['NamePrefix'] == 'Dr.':
                df = df.set_value(i, 'Age', 43.571)
            elif df.iloc[i]['NamePrefix'] == 'Major.':
                df = df.set_value(i, 'Age', 48.5)
            elif df.iloc[i]['NamePrefix'] == 'Col.':
                df = df.set_value(i, 'Age', 54.0)
            elif df.iloc[i]['NamePrefix'] == 'Mlle.':
                df = df.set_value(i, 'Age', 24.0)
    return df      
    
#Removes unneccessary features
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked', 'PassengerId', 'NamePrefix', 'SibSp', 'Parch'], axis=1)

#Feature engineers in preparation for encoding
def transform_features(df):
    df = format_name(df)
    simplify_fam(df)
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = drop_features(df)
    return df
from sklearn import preprocessing

#Encodes all features
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

#Feature engineers and encodes data
df_data = transform_features(df_data)
df_test = transform_features(df_test)
df_data, df_test = encode_features(df_data,df_test)

#Extracts numpy arrays from dataFrames to feed into neural network
data = df_data.iloc[:591,:].as_matrix()
dataCV = df_data.iloc[591:,:].as_matrix()
dataTest = df_test.as_matrix()




#Trains and evaluates neural network with sigOpt recommended values
def evaluate_model(assignments):
    layer_size = 50
    net = tfl.input_data(shape=[None, 6])
    net = tfl.fully_connected(net, layer_size,activation = 'relu')
    net = tfl.dropout(net,assignments['dropout'])
    net = tfl.fully_connected(net, layer_size,activation = 'relu')
    net = tfl.dropout(net, assignments['dropout'])
    net = tfl.fully_connected(net, layer_size,activation = 'relu')
    net = tfl.dropout(net, assignments['dropout'])
    net = tfl.fully_connected(net, 2, activation='softmax')
    net = tfl.regression(net, optimizer = tfl.optimizers.Adam (learning_rate=0.001, beta1=0.9, beta2=.999, epsilon=1e-08, use_locking=False, name='Adam'))
    
    model = tfl.DNN(net)
    model.fit(data, labels, n_epoch=assignments['epochs'], batch_size=16, show_metric=True)
    results = model.evaluate(data, labels)
    print('Training data accuracy: ' + str(results[0]))
    resultsCV = model.evaluate(dataCV, labelsCV)
    print('CV accuracy: ' + str(resultsCV[0]))
    print(resultsCV)
    tf.reset_default_graph()
    return resultsCV[0]

#Rounds neural network output to 1 or 0
def makePrediction(prediction, threshold):
    a = [0]*len(prediction)
    for i in range(len(prediction)):
        if prediction[i]< threshold:
            a[i] = 0
        else:
            a[i] = 1
    return a

'''
#Optimizes neural network with sigOpt
def optimize():
    conn.experiments(experiment.id).suggestions().delete()
    for _ in range(100):
        
        tf.reset_default_graph()
        suggestion = conn.experiments(experiment.id).suggestions().create()
        value = evaluate_model(suggestion.assignments)
        conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                value=value,
            )

#Clears observations from sigOpt experiment in prepartion for new optimization session
def clearObservations():
    conn.experiments(experiment.id).observations().delete()
'''
    
'''
Optimal values


.79904
exportPredictions(50,0.001,.9,.999,0.7,1000)


.78469
exportPredictions(52, 0.0014885321593006061, .9, .999, 0.40, 249)
exportPredictions(52, 0.001, .9, .999, 0.40, 249)


Beta1 ?
Beta2 .85-.9
Layer Size 20-70
Learning Rate .006-.03
Dropout .9-1
Epochs 100-300
exportPredictions(52, 0.0014885321593006061, .15, .7, 0.9, 249)

(layer_size, learning_rate, beta1, beta2)
exportPredictions(52, 0.014885321593006061, .15, .7, 0.962923647182921, 249) Epochs = 249, Dropout = 0.962923647182921
exportPredictions(20, 0.012949984921935807,0.17574611159754028,0.8,1, 200 )
exportPredictions(41, 0.0016451543576259664,0.1389314441775895,0.6233181098845707,1, 200)
'''

#Trains and evaluates neural network with given values. Also outputs predictions on test data
def exportPredictions(layer_size, learning_rate, beta1, beta2,dropout,epochs):
    tf.reset_default_graph()
    net = tfl.input_data(shape=[None, 6])
    net = tfl.fully_connected(net, layer_size,activation = 'relu')
    net = tfl.dropout(net, dropout)
    net = tfl.fully_connected(net, layer_size,activation = 'relu')
    net = tfl.dropout(net, dropout)
    net = tfl.fully_connected(net, layer_size,activation = 'relu')
    net = tfl.dropout(net, dropout)
    net = tfl.fully_connected(net, 2, activation='softmax')
    net = tfl.regression(net, optimizer = tfl.optimizers.Adam (learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08, use_locking=False, name='Adam'))


    model = tfl.DNN(net)
    model.fit(data, labels, n_epoch=epochs, batch_size=16, show_metric=True)
    results = model.evaluate(data, labels)
    print('Training data accuracy: ' + str(results[0]))
    resultsCV = model.evaluate(dataCV, labelsCV)
    print('CV accuracy: ' + str(resultsCV[0]))
    testPredictPerc = model.predict(dataTest)
    testPredictPerc = np.delete(testPredictPerc, 0,1)
    testPredict = makePrediction(testPredictPerc, 0.5)
    df = pd.DataFrame(testPredict)
    df.index = range(892,len(df)+892)
    df.columns = ['Survived']
    df.index.names = ['PassengerId']
    
    df.to_csv(path_or_buf = 'predictions.csv', sep=',')
    
