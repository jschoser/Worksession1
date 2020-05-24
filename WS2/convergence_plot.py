from matplotlib import pyplot as plt
import numpy as np

n = np.array([1, 2, 4, 8])
RMS = np.array([0.016837104826, 0.0048154109796, 0.0012914192985, 0.0003299453713174282])
n = np.log(1/n)
RMS = np.log(RMS)
plt.plot(n, RMS, marker='o')
slope = (RMS[0] - RMS[-1]) / (n[0] - n[-1])
print(slope)
plt.show()
