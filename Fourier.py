import numpy as np
import matplotlib.pyplot as plt

Fs = 256  # 采样频率 要大于信号频率的两倍可恢复信号，有了采样频率我们可以知道原信号一个周期内可以采样多少个点，即FS/F
t = np.arange(0, 1, 1.0/Fs)  # 1s采样Fs个点

F1 = 50  # 信号1的频率
F2 = 75  # 信号2的频率
y = 2 + 3*np.cos(2*np.pi*F1*t+np.pi/3) + 1.5*np.cos(2*np.pi*F2*t)

N = len(t)  # 采样点数

freq = np.arange(N) / N * Fs   #也就是说，傅里叶分解后能观测到的频域有什么频率是由采样点个数和采样频率得到的
freq2 = np.fft.fftfreq(len(t), 1.0/Fs)     #等价于freq
print(freq2)


Y1 = np.fft.fft(y)   # 复数

Y = Y1 / (N/2)  # 换算成实际的振幅
Y[0] = Y[0] / 2

freq_half = freq[range(int(N/2))]
Y_half = Y[range(int(N/2))]

fig, ax = plt.subplots(5, 1, figsize=(12, 12))
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
plt.show()
#plt.savefig('a.png')
#plt.close()
#相位谱怎么求 arctan(虚部除以实部)
print(Y[50].real)
print(Y[50].imag)
print(np.arctan2(Y[50].imag,Y[50].real)/np.pi)
