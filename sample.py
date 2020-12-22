import numpy as np
from math import sqrt, pi, exp
import matplotlib.pyplot as plt

N = 4096            # サンプル数
s = N/256           # 標準偏差

y = []
for i in range(N):
  x = i - N/2
  v = exp(-x**2/(2.0*s**2))/(sqrt(2*pi)*s)
  y.append(v)


fk = np.fft.fft(y)
# plt.plot(fk)
# plt.show()

freq = np.fft.fftfreq(N)
plt.xlim([-0.05,0.05])
plt.plot(freq,np.abs(fk))
plt.show()