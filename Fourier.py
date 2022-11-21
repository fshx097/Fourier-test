import numpy as np
import matplotlib.pyplot as plt

Fs = 256  # 采样频率 要大于信号频率的两倍可恢复信号，有了采样频率我们可以知道原信号一个周期内可以采样多少个点，即FS/F
t = np.arange(0, 1, 1.0/Fs)  # 1s采样Fs个点

F1 = 50  # 信号1的频率
F2 = 75  # 信号2的频率
F3 = 21  # 信号2的频率
F4 = 100  # 信号2的频率
F5 = 68  # 信号2的频率
F6 = 500  # 信号2的频率
F7 = 2  # 信号2的频率
F8 = 33  # 信号2的频率
F9 = 145  # 信号2的频率
F10 = 90  # 信号2的频率

y = 2 + 3*np.cos(2*np.pi*F1*t+np.pi/3) + 1.5*np.cos(2*np.pi*F2*t) + 3*np.cos(2*np.pi*F3*t+np.pi/6) + 1.5*np.cos(2*np.pi*F4*t) + 3*np.cos(2*np.pi*F5*t+np.pi/8) + 2*np.cos(2*np.pi*F6*t) + 3*np.cos(2*np.pi*F7*t+np.pi/3) + 4.5*np.cos(2*np.pi*F8*t) + 3.5*np.cos(2*np.pi*F9*t+np.pi/4) + 1.5*np.cos(2*np.pi*F10*t)


N = len(t)  # 采样点数

freq = np.arange(N) / N * Fs   #也就是说，傅里叶分解后能观测到的频域有什么频率是由采样点个数和采样频率得到的
freq2 = np.fft.fftfreq(len(t), 1.0/Fs)     #等价于freq
print(freq2)


Y1 = np.fft.fft(y)   # 复数

Y = Y1 / (N/2)  # 换算成实际的振幅
Y[0] = Y[0] / 2

f=[]
for i in range(Y.shape[0]):
    if Y[i] > 0.1:
        f.append(i)
print(f"检测到频率个数为{len(f)}")

freq_half = freq[range(int(N/2))]
Y_half = Y[range(int(N/2))]

fig, ax = plt.subplots(6, 1, figsize=(12, 12))
ax[0].plot(t, y)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')

ax[1].plot(freq, abs(Y1), 'r', label='no normalization')
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('Amplitude')
ax[1].legend()

ax[2].plot(freq, abs(Y), 'r', label='normalization')
ax[2].set_xlabel('Freq (Hz)')
ax[2].set_ylabel('Amplitude')
ax[2].set_yticks(np.arange(0, 3))
ax[2].legend()

ax[3].plot(freq_half, abs(Y_half), 'b', label='normalization')
ax[3].set_xlabel('Freq (Hz)')
ax[3].set_ylabel('Amplitude')
ax[3].set_yticks(np.arange(0, 3))
ax[3].legend()
#傅里叶逆变换
iy = np.fft.ifft(Y1).real
ax[4].plot(t, iy)
ax[4].set_xlabel('Time (s)')
ax[4].set_ylabel('Amplitude')


#对傅里叶分解的结果，我们只选择一些频率进行逆变换，可以得到原序列的相似序列，因此通过仔细选择记录哪些系数，我们可以执行压缩、去噪等多种任务。
Y2 = Y1.copy()
for i in range(5):
    Y2[int(f[i])] = 0
iy = np.fft.ifft(Y2).real
ax[5].plot(t, iy)
ax[5].set_xlabel('Time (s)')
ax[5].set_ylabel('Amplitude')
plt.show()

#plt.savefig('a.png')
#plt.close()
#相位谱怎么求 arctan(虚部除以实部)
print(Y[50].real)
print(Y[50].imag)
print("相位谱：",np.arctan2(Y[50].imag,Y[50].real)/np.pi)
