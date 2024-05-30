import librosa
import librosa.feature
import numpy as np
from playsound import playsound
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.fftpack import dct


#音频读取，波形时间序列+采样率
audio_data = 'C:/Users/26514/Desktop/Musical Instruments/T0323/2.wav'
original_signal , sample_rate = librosa.load(audio_data)
print(original_signal.shape, sample_rate)
print('audio:', original_signal)
print('audio shape:', np.shape(original_signal))
print('Sample Rate (KHz):', sample_rate)
print('Check Len of Audio:', np.shape(original_signal)[0]/sample_rate)
#播放音频
playsound(audio_data)
# 绘制音频波形图
plt.figure(figsize=(10, 6))
librosa.display.waveshow(original_signal, sr=sample_rate)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
# 绘制音频频谱图
plt.figure(figsize=(10, 6))
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(original_signal), ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()
#音频增强：增强高频
pre_emphasis = 0.97
emphasized_signal = np.append(original_signal[0], original_signal[1:] - pre_emphasis * original_signal[:-1])
emphasized_signal_num = np.arange(len(emphasized_signal))
#音频增强：不变
# emphasized_signal = original_signal
# emphasized_signal_num = np.arange(len(original_signal))
print(emphasized_signal_num)
# 分帧
frame_size = 0.25
frame_stride = 0.01
frame_length = int(round(frame_size*sample_rate))
frame_step = int(round(frame_stride*sample_rate)) 
signal_length = len(emphasized_signal)
num_frames = int(np.ceil(float(np.abs(signal_length-frame_length))/frame_step))
pad_signal_length = num_frames * frame_step + frame_length
pad_signal = np.append(emphasized_signal, np.zeros((pad_signal_length - signal_length)))
indices = np.tile(np.arange(0,frame_length),(num_frames,1))+np.tile(np.arange(0,num_frames*frame_step,frame_step), (frame_length, 1)).T
frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

# 汉明窗
N = 200
frames *= np.hamming(frame_length)

# 傅里叶变换和功率谱
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = (1.0 / NFFT) * (mag_frames ** 2)

# 将频率转换为Mel频率
low_freq_mel = 0

nfilt = 40
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right
    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB

num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
(nframes, ncoeff) = mfcc.shape
mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
print(mfcc.shape)

#MFCC热度图
plt.figure(figsize=(11,7), dpi=500)

plt.subplot(211)
plt.imshow(np.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.1, extent=[0,filter_banks.shape[1],0,filter_banks.shape[0]]) #热力图
plt.title("MFCC")

plt.subplot(212)
plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.1, extent=[0,mfcc.shape[0],0,mfcc.shape[1]]) #热力图
plt.title("MFCC")

plt.show()
