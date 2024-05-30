import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import librosa
import soundfile as sf

def preprocess(file_path,save_path,db_threhold,output_name='remix.wav'):
    # 读取输入目录下的所有wav文件
    # file_path = "/Users/26514/Desktop/Musical Instruments/Sample"
    # 存放所有文件名
    p = Path(file_path)
    # 所有以wav结尾的文件
    count = 0
    data=[]
    # 将所有音频数据链接到数组，因为数据特性，不同音乐之间存在静音区间，可以不考虑突变
    for file_name in p.rglob('*.wav'):
        audio=file_name
        count =count+1
        aduio_data , sr = librosa.load(audio)
        print (aduio_data)
        data=np.append(data,aduio_data)
    print (data.shape)
    # 获取静音区间，阈值为输入阈值
    intervals = librosa.effects.split(data,top_db=db_threhold)
    # 去除静音区间
    data_remix = librosa.effects.remix(data,intervals)
    librosa.display.waveshow(data)
    librosa.display.waveshow(data_remix)
    # 保存合并去除静音后数据
    save_path = file_path + '/' + output_name
    sf.write(save_path, data_remix, samplerate=sr)
    # 输出到保存路径，保存为采样率不变的一个wav文件
    print('Done.')

