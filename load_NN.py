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

# Z正则化
def Z_ScoreNormalization(x):
    mean_v = np.mean(x)
    std_v = np.std(x)
    for i in x:
        i = (i - mean_v) / std_v
    return x

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
def NFFT(frames,frame_len,hop_len,k = 1,plot = 0):
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
def mel_filter(pow_frame,sr,NFFT,plot = 0,N = 40):
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

# 输入音频数据
def import_data(file):
    audio, sr = librosa.load(file)
    audio = Z_ScoreNormalization(audio)
    # print(audio)
    frames , frame_len , hop_len = enframe(audio, sr , 50 , 25)
    # print(frames.shape)
    # print(frames)
    pow_frames = NFFT(frames,frame_len,hop_len,1,0)
    # print(pow_frames.shape)
    # print(pow_frames)
    filter_banks=mel_filter(pow_frames,sr,frame_len,0,40)
    # print(filter_banks)
    # print(filter_banks.shape)
    return filter_banks

# 预测函数，输入需要辨别的音频目录和使用的NN目录，返回
def predict_fuction(file_path,model_path):
    nn_model = tf.keras.models.load_model(model_path)
    aduio_data , sr = librosa.load(file_path)
    intervals = librosa.effects.split(aduio_data,top_db=10)
    # 去除静音区间
    test_audio = librosa.effects.remix(aduio_data,intervals)
    # print(test_audio.shape[0],'\t',test_audio.shape[1])
    lenth = test_audio.shape[0]
    predictions=np.zeros[lenth]
    for i in range(0,lenth):
        predictions[i] = nn_model.predict_classes(test_audio[i])
    counts = np.bincount(predictions)
    instrument_pre=np.argmax(counts)
    result = np.sum(predictions == instrument_pre)
    trust = result/lenth
    return instrument_pre,trust

# # Save the entire model as a `.keras` zip archive.

def main(file_path,model_path='./model.keras')
    instrument_pre, trust = predict_fuction(file_path,model_path)
    return instrument_pre, trust
