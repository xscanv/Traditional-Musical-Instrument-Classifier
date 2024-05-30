import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn import svm

def data_create(filter_banks,label):
    num_of_row = filter_banks.shape[0]
    label_data = []
    for i in range(num_of_row):
        label_data.append(label)
    x_train, x_test, y_train, y_test = train_test_split(filter_banks, label_data, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def dataset_create():
    all_x_train = ()
    all_x_test = ()
    all_y_train = ()
    all_y_test = ()
    return all_x_train,all_x_test,all_y_train,all_y_test

def dataset_append(all_x_train , all_x_test , all_y_train,all_y_test,x_train, x_test, y_train, y_test):
    all_x_train = all_x_train + x_train
    all_x_test = all_x_test + x_test
    all_y_train = all_y_train + y_train
    all_y_test = all_y_test +y_test
    return all_x_train,all_x_test,all_y_train,all_y_test

def training_svm(all_x_train,all_x_test,all_y_train,all_y_test):

    svm_model = svm.SVC(C=1, kernel='linear')
    svm_model.fit(all_x_train,all_y_train)
    print("The accuracy on training data is %s" % svm_model.score(all_x_train,all_y_train))
    print("The accuracy on test data is %s" % svm_model.score(all_x_test,all_y_test))
    return svm_model

def classify_mel(all_x_test,all_y_test):
    y_pred = svm_model.predict(all_x_test)

    accuracy = accuracy_score(all_y_test, y_pred)
    print(f"模型准确率: {accuracy:.2f}")
