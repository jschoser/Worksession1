import numpy as np

f1 = 148.06161051975872
f2 = 147.81398506366472
f3 = 147.7509524718592
r = 0.5

ooa = np.log((f3 - f2) / (f2 - f1)) / np.log(r)
e_int = (f2 - f3) / (r ** 2 - 1)

print("ooa", ooa)
print("e", e_int)
