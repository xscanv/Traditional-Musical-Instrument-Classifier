import numpy as np
import pandas as pd

def create_list(file_path):
    file = pd.read_excel(file_path)
    row_num = file.shape[0]
    col_num = file.shape[1]
    print("行数：", row_num)
    print("列数：", col_num)
    dictionary = np.zeros((row_num, col_num), dtype=object)
    for i in range(row_num):
        for j in range(col_num):
            dictionary[i, j] = file.at[i, file.columns[j]]
    return dictionary,row_num,col_num

#file_path = 'C:/Users/Lenovo/Desktop/instrument list.xlsx'
#dictionary = create_list(file_path)
#print(dictionary)
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import librosa
import soundfile as sf
from pathlib import Path
import multiprocessing

def divide_preprocess(number,load_root_file_path,dictionary,save_root_file_path,db_threhold,output_name):
    # 读取输入目录下的所有wav文件
    # file_path = "/Users/26514/Desktop/Musical Instruments/Sample"
        i=number
        print(i," is starting.")
        load_file_path=load_root_file_path+'/'+dictionary    # 创建一级输入分录的路径
        my_file = Path(load_file_path)
        if my_file.exists():
            # 指定的文件或目录存在
            save_file_path = save_root_file_path + '/' + dictionary+output_name    # 创建一级输出分录的路径
            save_file=Path(save_file_path)
            if not (save_file.exists()):
                #整合音频还没有输出
                p = Path(load_file_path)
                # 一级输入分录下所有以wav结尾的文件
                data=[]
                # 将所有音频数据链接到数组，因为数据特性，不同音乐之间存在静音区间，可以不考虑突变
                for file_name in p.rglob('*.wav'):
                    audio=file_name
                    aduio_data , sr = librosa.load(audio)
                    data=np.append(data,aduio_data)
                # 获取静音区间，阈值为输入阈值
                intervals = librosa.effects.split(data,top_db=db_threhold)
                # 去除静音区间
                data_remix = librosa.effects.remix(data,intervals)
                librosa.display.waveshow(data)
                librosa.display.waveshow(data_remix)
                # 保存合并去除静音后数据
                sf.write(save_file_path, data_remix, samplerate=sr)
                # 输出到保存路径，保存为采样率不变的一个wav文件
        print(i," is done.")

if __name__ == '__main__':
    file_path = 'C:/Users/Lenovo/Desktop/instrument list.xlsx' #请输入”乐器目录的文件地址“
    dictionary, row_num, col_num = create_list(file_path) #返回乐器目录、行、列，其中272行、2列
    print(dictionary)
    load_root_file_path = 'C:/Users/Lenovo/Desktop/10.中国传统乐器音响数据库（CTIS）/中国传统乐器音响数据库（CTIS）/民族乐器数据库' #请输入“音乐文件的目录的目录（包含所有乐器的那层目录）”
    save_root_file_path = 'C:/Users/Lenovo/Desktop/remix2' #请输入“你想储存整合后文件的目录（整合音频保存在哪里）”
    db_threshold = 10#请输入“你认为多少分贝以下算‘静音’”
    output_name = '.wav'#整合后音频文件的后缀，可自行更改，但一定要是.wav
    pool = multiprocessing.Pool()#创建线程池
    results = pool.starmap(divide_preprocess, [(i,load_root_file_path, dictionary[i][0], save_root_file_path, db_threshold, output_name) for i in range(row_num)])#并发运行线程池
    pool.close()
    pool.join()
    print("所有进程已结束")#应该有220个整合音频，如果是218个，那么可能是L0293、L0307没有保存下来，请点进原文件夹，删除其中非“.wav”的文件