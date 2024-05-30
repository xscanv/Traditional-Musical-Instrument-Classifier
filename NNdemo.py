import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn import svm
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# 打开MUSICAL iNSTRUMENTS 文件夹并运行

# 获取目录下所有文件
def get_file_names(directory):
    file_names = os.listdir(directory)
    return file_names

# 读取xlsx列表
def create_list(file_path):
    file = pd.read_excel(file_path)
    row_num = file.shape[0]
    col_num = file.shape[1]
    # print("行数：", row_num)
    # print("列数：", col_num)
    dictionary = np.zeros((row_num, col_num), dtype=object)
    for i in range(row_num):
        for j in range(col_num):
            dictionary[i, j] = file.at[i, file.columns[j]]
    return dictionary,row_num,col_num

# 分帧函数
def enframe(audio , sr , frame_t , hop_t):
    # 传入参数：单通道音频序列，采样率，帧时长，步进时长
    frame_len = int(frame_t / 1000 * sr)
    # print(frame_len)
    hop_len = int(hop_t / 1000 * sr)
    # print(hop_len)
    audio_len=len(audio) 
    # 计算信号总长度
    if audio_len <= frame_len: 
        frame_num = 1
    else:
        frame_num = int(np.ceil(1.0 * audio_len - frame_len + hop_len)/hop_len)
    #计算分帧帧数
    pad_length = int((frame_num) * hop_len + frame_len)
    # 计算所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length-audio_len))
    pad_audio = np.concatenate((audio,zeros))
    # Padding，用0补全残缺帧
    indices = np.tile(np.arange(0, frame_len), (frame_num, 1)) + np.tile(np.arange(0, frame_num *  hop_len, hop_len), (frame_len, 1)).T
    # 所有帧的时间点进行抽取，得到frame_num*frame长度的矩阵，tile() 函数将原矩阵复制展开
    indices = np.array(indices)  
    # 将indices转化为矩阵
    frames = pad_audio[indices]  
    # 得到帧信号矩阵
    frames *= np.hamming(frame_len) 
    # 加hamming window，可以对窗口种类进行调整
    return frames , frame_len , hop_len
    # 返回整体帧信号矩阵

# 快速傅里叶变换
def NFFT(frames,frame_len,hop_len,k = 1,plot = 1):
    NFFT = int(frame_len * k) 
    # 计算frame_length，k为调节参数，k越大信号损失越大
    mag_frames = np.absolute(np.fft.rfft(frames,NFFT))
    pow_frames = mag_frames**2/NFFT
    # 绘制响度图像
    if plot:
        plt.figure(dpi=300,figsize=(12,6))
        plt.imshow(20*np.log10(pow_frames[40:].T),cmap=plt.cm.jet,aspect='auto')
        plt.yticks([0,128,256,384,512],np.array([0,128,256,384,512])*sr/NFFT)
        plt.show()
    return pow_frames

# mel滤波器，输出filter_bnaks, 
def mel_filter(pow_frame,sr,NFFT,plot=1,N=40):
    # 定义mel filter
    mel_N = N
    # 定义滤波器数量
    mel_low, mel_high = 0, (2595*np.log10(1+(sr/2)/700))
    mel_freq = np.linspace(mel_low,mel_high,mel_N+2)
    hz_freq = (700 * (10**(mel_freq / 2595) - 1))
    bins = np.floor((NFFT)*hz_freq/sr) 
    # 将频率转换成对应的sample位置
    fbank = np.zeros((mel_N,int(NFFT/2+1))) 
    # 每一行储存一个梅尔滤波器的数据
    for m in range(1, mel_N + 1):
        f_m_minus = int(bins[m - 1])   # left
        f_m = int(bins[m])             # center
        f_m_plus = int(bins[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    filter_banks = np.matmul(pow_frame, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  
    # np.finfo(float)是最小正值
    filter_banks = 20 * np.log10(filter_banks)  # dB
    filter_banks -= np.mean(filter_banks,axis=1).reshape(-1,1) # 全幅归一化
    # 绘制示意图
    if plot:
        plt.figure(dpi=300,figsize=(12,6))
        plt.imshow(filter_banks[40:].T, cmap=plt.cm.jet,aspect='auto')
        plt.yticks([0,10,20,30,39],[0,1200,3800,9900,22000])
        plt.show()
    return filter_banks

# 由处理的filter_banks创建数据集
def data_create(filter_banks,label,test_size=0.001):
    num_of_row = filter_banks.shape[0]
    label_data = []
    for i in range(num_of_row):
        label_data.append(label)
    x_train, x_test, y_train, y_test = train_test_split(filter_banks, label_data, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

# 创建空数据集
def dataset_init():
    all_x_train = ()
    all_x_test = ()
    all_y_train = ()
    all_y_test = ()
    return all_x_train,all_x_test,all_y_train,all_y_test

# 向数据集添加乐器
def dataset_append(all_x_train , all_x_test , all_y_train,all_y_test,x_train, x_test, y_train, y_test):
    if len(all_x_train):
        all_x_train = np.concatenate((all_x_train, x_train),axis=0)
        all_x_test = np.concatenate((all_x_test , x_test),axis=0)
        all_y_train = np.concatenate((all_y_train, y_train),axis=0)
        all_y_test = np.concatenate((all_y_test , y_test),axis=0)
    else:
        all_x_train =  x_train
        all_x_test =  x_test
        all_y_train = y_train
        all_y_test =  y_test
    return all_x_train,all_x_test,all_y_train,all_y_test

# SVM训练
def training_svm(all_x_train , all_y_train):
    svm_model = svm.SVC(C=1, kernel='linear',max_iter=100)
    svm_model.fit(all_x_train,all_y_train)
    svm_score = svm_model.score(all_x_train,all_y_train)
    # print("The accuracy on training data is %s" % svm_model.score(all_x_train,all_y_train))
    return svm_model,svm_score

# 检验SVM模型准确度
def classify_mel(all_x_test,all_y_test,svm_model):
    accuracy = svm_model.score(all_x_test,all_y_test)
    return accuracy

# 随机选择两个小于len的不同随机数，返回1*2的列表
def create_ran(len):
    s1 = random.randint(0,len-1)
    s2 = random.randint(0,len-1)
    while(s1==s2):
        print('HIT!')
        s2=random.randint(0,len-1)
    chosen=[s1,s2]
    return chosen

#  创建随机序号表，上限为lenth，选取n个不同的随机数
def create_rans(lenth,n):
    start_range = 0
    end_range = lenth
    all_numbers = list (range(start_range, end_range))
    random.shuffle(all_numbers)
    random_numbers = all_numbers[:n]
    return random_numbers

#  对传入的list进行Z标准化
def Z_ScoreNormalization(x):
    mean_v = np.mean(x)
    std_v = np.std(x)
    for i in x:
        i = (i - mean_v) / std_v
    return x

#  根据输入的序号表，选择对应的乐器编号音频创建数据库
def create_train_data(num_list):
    X_train , X_test , Y_train , Y_test = [],[],[],[]
    for i in num_list:
        audio, sr = librosa.load('./remix/' + files_list[i] + '.wav')
        audio = Z_ScoreNormalization(audio)
        # print(audio)
        frames , frame_len , hop_len = enframe(audio, sr , 50 , 25)
        # print(frames.shape)
        # print(frames)
        pow_frames = NFFT(frames,frame_len,hop_len,1,0)
        # print(pow_frames.shape)
        # print(pow_frames)
        filter_banks=mel_filter(pow_frames,sr,frame_len,0,mel)
        # print(filter_banks)
        # print(filter_banks.shape)
        x_train, x_test, y_train, y_test = data_create(filter_banks,i,0.2)
        X_train , X_test , Y_train , Y_test = dataset_append(X_train , X_test , Y_train , Y_test , x_train , x_test, y_train, y_test)
    return(X_train,X_test,Y_train,Y_test)

#  根据输入的序号表，选择对应的乐器编号音频创建序列数据库
def create_LSTM(num_list):
    X_train , X_test , Y_train , Y_test = [],[],[],[]
    for i in num_list:
        audio, sr = librosa.load('./remix/' + files_list[i] + '.wav')
        audio = Z_ScoreNormalization(audio)
        # print(audio)
        frames , frame_len , hop_len = enframe(audio, sr , 50 , 25)
        # print(frames.shape)
        # print(frames)
        pow_frames = NFFT(frames,frame_len,hop_len,1,0)
        # print(pow_frames.shape)
        # print(pow_frames)
        filter_banks=mel_filter(pow_frames,sr,frame_len,0,mel)
        # print(filter_banks)
        # print(filter_banks.shape)
        x_train, x_test, y_train, y_test = data_create(filter_banks,i,0.2)
        X_train , X_test , Y_train , Y_test = dataset_append(X_train , X_test , Y_train , Y_test , x_train , x_test, y_train, y_test)
    return(X_train,X_test,Y_train,Y_test)

def term_create(filter_banks,label,width,test_size=0.001):
    num_of_row = filter_banks.shape[0]
    num_of_matrix = num_of_row-width
    label_data = []
    matrix_data = []
    for i in range(num_of_matrix):
        label_data.append(label)
        matrix=filter_banks[i,i+width]
        matrix_data.append(matrix)
    x_train, x_test, y_train, y_test = train_test_split(matrix_data, label_data, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

    

#  创建CNN网络，并根据传入的训练集进行训练，传出模型并显示准确率得分情况
def CNN(X_train,X_test,Y_train,Y_test,mel,epochs,batch_size_timer):
    # define model
    output_num = len(np.unique(Y_train))
    if (len(np.unique(Y_test))>output_num):
        output_num = len(np.unique(Y_test))
    
    input_x = tf.keras.Input(shape=(mel,))
    hidden1 = layers.Dense(128, activation='relu')(input_x)
    hidden2 = layers.Dense(128, activation='relu')(hidden1)
    hidden3 = layers.Dense(128, activation='relu')(hidden2)
    pred = layers.Dense(220, activation='softmax')(hidden3)

    model = tf.keras.Model(inputs=input_x, outputs=pred)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size_timer*1024, verbose=1)
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

    # evaluate the model on training set and test set
    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    print('Test Accuracy on the training set: %.3f' % train_acc)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test Accuracy on the test set: %.3f' % test_acc)
    model.summary()
    return model


#  创建对时间序列矩阵的卷积CNN网络，并根据传入的训练集进行训练，传出模型并显示准确率得分情况
def ICNN(X_train,X_test,Y_train,Y_test,mel,width,epochs,batch_size_timer):
    # define model
    output_num = len(np.unique(Y_train))
    if (len(np.unique(Y_test))>output_num):
        output_num = len(np.unique(Y_test))
    
    input_x = tf.keras.Input(shape=(mel,width))
    layers_Cov = tf.keras.layers.Conv2D(64,(3,3),padding='same')(input_x)
    layers_Flatten = tf.keras.layers.Flatten(input_shape=[mel,width])(layers_Cov)
    hidden1 = layers.Dense(128, activation='relu')(layers_Flatten)
    hidden2 = layers.Dense(128, activation='relu')(hidden1)
    hidden3 = layers.Dense(128, activation='relu')(hidden2)
    pred = layers.Dense(220, activation='softmax')(hidden3)

    model = tf.keras.Model(inputs=input_x, outputs=pred)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size_timer*1024, verbose=1)
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

    # evaluate the model on training set and test set
    train_loss, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    print('Test Accuracy on the training set: %.3f' % train_acc)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test Accuracy on the test set: %.3f' % test_acc)
    model.summary()
    return model

#  预处理模型的YLABEL，将数值标签转化为one_hot编码向量
def one_hot(labels, num_classes=None):
    if num_classes is None:
        num_classes = 220
    return np.eye(num_classes)[labels]

#  创建用于测试单一乐器准确度的测试用数据集，输入files_list序号，返回数据集
def create_test_data(num):
    X_test , Y_test = [],[]
    audio, sr = librosa.load('./remix/' + files_list[num] + '.wav')
    audio = Z_ScoreNormalization(audio)
    # print(audio)
    frames , frame_len , hop_len = enframe(audio, sr , 50 , 25)
    # print(frames.shape)
    # print(frames)
    pow_frames = NFFT(frames,frame_len,hop_len,1,0)
    # print(pow_frames.shape)
    # print(pow_frames)
    filter_banks=mel_filter(pow_frames,sr,frame_len,0,mel)
    # print(filter_banks)
    # print(filter_banks.shape)
    X_train, X_test, Y_train, Y_test = data_create(filter_banks,num,0.8)
    return(X_test,Y_test)

#  测试具体种类的音乐的预测准确率，输入对应乐器在namelist中的序号与CNN模型，返回单一乐器的test_loss,test_acc
def testing(num,model):
    X_test,Y_test=create_test_data(num)
    Y_test_H = one_hot(Y_test)+np.zeros(220)
    test_l, test_a = model.evaluate(X_test, Y_test_H, verbose=0)
    return test_l,test_a

epochs = 10
mel = 40
batch_size_timer = 8
# print(os.getcwd())
list_path = './instrument list.xlsx' 
# 已修改为相对地址
remix_list = get_file_names('./remix.')
files_list = []
for name in remix_list:
    files_list.append(name[0:5])
# print(len(name_list))
dictionary, row_num, col_num = create_list(list_path)
X_train, X_test, Y_train, Y_test = dataset_init()
lenth = len(files_list)
num_list = create_rans(lenth,lenth)
X_train,X_test,Y_train,Y_test = create_train_data(num_list)
Y_train_H = one_hot(Y_train)
Y_test_H = one_hot(Y_test)
model = CNN(X_train,X_test,Y_train_H,Y_test_H,mel,epochs,batch_size_timer)

score_list=[]
name_list=[]
for num in range(0,220):
    test_l,test_a = testing(num,model)
    score_list.append(test_a)
    name_list.append(files_list[num])
    data = {'No.':name_list,'Accuracy':score_list}
    df = pd.DataFrame(data)
df.to_csv('./instrument_score.csv')