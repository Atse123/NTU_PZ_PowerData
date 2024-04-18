# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 02:15:36 2023

@author: q0389
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cProfile import label
from scipy import signal
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cProfile import label
from scipy import signal
import math
from scipy.optimize import curve_fit 
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import sympy as sp


# Calculate power

HZ=60
samplerate =12500
sampleperiod =1/samplerate
T=1/HZ

df=pd.read_table('VP-6_modeldata.txt', sep=',')
df.iloc[:,[3]] = df.iloc[:,[3]]*100
df.iloc[:,[4]] = df.iloc[:,[4]]*100
W1 =df["V1"]* df["I1"]
W1 = np.array(W1)
W2 =df["V2"]* df["I2"]
W2 = np.array(W2)
W=(W1+W2)*sampleperiod*HZ
dx = pd . DataFrame(W)
num_rows_len = len(df)
m=num_rows_len/208
wn=2*2/12500
b,a =signal.butter(4,wn,'low')
filtedData = signal.filtfilt(b,a,dx.iloc[:,0].values)
c=np.array_split(filtedData,m)
de = pd . DataFrame(c)
de['sum']=de.apply(lambda x:x.sum(), axis=1)
de['sum']=de['sum']


de = de.drop(range(930))
de.reset_index(drop=True, inplace=True)
plt.plot(de["sum"])

# %%  standby power model

# standby power
base=540

print('P_stadby=540 [W]')

#%% spindle power model

Sparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]

S=de.iloc[[200,500,800,1100,1400,1700,2000,2320,2600,2900,3200,3500,3800,4100,4400,4700,5000,5300,5600,5900], 209].values
Sp=S-base

x_data = Sparam
y_data = Sp
Spd = pd.DataFrame(Sp)




x1 = np.array([500, 1000, 1500])
y = np.array(Spd.iloc[[0, 1, 2], 0].values)


def func(X, a, b): 
    y = a * X + b 
    return y
from scipy.optimize import curve_fit
popt, pcov = curve_fit(func, x1, y)
a_fit = popt[0]
b_fit = popt[1]





x1 = np.array([1500,2500,3000,3500])
y = np.array(Spd.iloc[[2,4,5,6], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y

from scipy.optimize import curve_fit
popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]




x1 = np.array([4000,6000,8000,10000])
y = np.array(Spd.iloc[[7,11,15,19], 0].values)

def func(X, a, b,c,d): 
    y = a*X**3+b*X**2+ c*X+d  
    return y

from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]
c_fit = popt[2]
d_fit = popt[3]



data_points = [(500, Sp[0]), (1000, Sp[1]),(1500, Sp[2]) ]
data_points1 = [(1500, Sp[2]), (2000, Sp[3]), (2500, Sp[4]), (3000, Sp[5]), (3500, Sp[6]),(4000, Sp[7])  ]
data_points2 = [(4000, Sp[7]),(4500, Sp[8]),(5000, Sp[9]),(5500, Sp[10]),(6000, Sp[11]), (6500, Sp[12]), (7000, Sp[13]) , (7500, Sp[14]), (8000, Sp[15]), (8500, Sp[16]), (9000, Sp[17]), (9500, Sp[18]), (10000, Sp[19])]


def equation(x, m, c):
    return m * x + c


def equation1(x1, m1, c1):
    return m1 *x1+c1

def equation2(x2, c2,v2,b2,n2):
    return c2 * x2**3 + v2 * x2**2+b2*x2+n2


m = 0.14
c = 30.2

m1 = -0.04
c1 = 293.42


c2 = - 1.17525533997188e-9
v2 = 3.44343516317154e-5
b2 = - 0.322934401184431
n2 = 1342.63047066526


x_data = [point[0] for point in data_points]
y_data = [point[1] for point in data_points]

x_data1 = [point[0] for point in data_points1]
y_data1 = [point[1] for point in data_points1]

x_data2 = [point[0] for point in data_points2]
y_data2 = [point[1] for point in data_points2]



plt.figure(figsize=(8, 5), dpi=160, linewidth=2.25)  # 寬度為 16，長度為 10

plt.scatter(x_data, y_data, color='red',s=150)
plt.scatter(x_data1, y_data1, color='orange',s=150)
plt.scatter(x_data2, y_data2, color='green',s=150)



x_values = list(range(min(x_data), max(x_data) + 1))
y_values = [equation(x, m, c) for x in x_values]
plt.plot(x_values, y_values, color='red',linewidth=3)

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation1(x1, m1, c1) for x1 in x_values]
plt.plot(x_values, y_values, color='orange',linewidth=3)

x_values = list(range(min(x_data2), max(x_data2) + 1))
y_values = [equation2(x2,  c2, v2, b2, n2) for x2 in x_values]
plt.plot(x_values, y_values, color='green',linewidth=3)


custom_x_ticks = [ 2500, 5000, 7500, 10000]
plt.xticks(custom_x_ticks)

custom_y_ticks = [ 0, 200, 400, 600]
plt.yticks(custom_y_ticks)


plt.xticks(fontsize=28, fontname='Arial')
plt.yticks(fontsize=28,  fontname='Arial')


plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)


plt.tick_params(axis='both', direction='in', width=2, pad=6,length=6)

plt.xlabel('Spindle [rpm]', fontsize=33, fontname='Arial')
plt.ylabel('Power [W]', fontsize=33, fontname='Arial')

plt.legend().set_visible(False)
plt.grid(False)
plt.title('VP-6',fontsize=28, fontname='Arial')

plt.savefig('spindle_power.svg', format='svg', transparent=True, bbox_inches='tight')

plt.show()


print('P_spindle=30.2+0.14n  Spindle speed Range500-1500')
print('P_spindle=293.42-0.04n  Spindle speed Range1500-4000')
print('P_spindle=1342.63-0.32n+(3.44*10**-5)n**2+(-1.18*10**-9)n**3  Spindle speed Range4000-10000')

# %%feed power model


Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]

F=de.iloc[[6600,6800,7000,7200,7400,7600,7800,8000,8300,8800,9000,9200,9400,9600,9800,9950], 209].values
Fpx=F-de.iloc[[6225], 209].values
Fp = pd.DataFrame(Fpx)


x1 = np.array([500,4000,7500])
y = np.array(Fp.iloc[[0,  7,14], 0].values)

def func(X, a, b): 
    y = a * X + b  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)

a_fit = popt[0]
b_fit = popt[1]




Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]

F=de.iloc[[10600,11000,11600,11800,11500,11800,12100,12300,12500,12700,12900,13150,13300,13550,13700,13875], 209].values
Fpy=F-de.iloc[[10160], 209].values
Fp= pd.DataFrame(Fpy)

x1 = np.array([500,4000,7500])
y = np.array(Fp.iloc[[0, 7,14], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y

from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]



# Fzu
Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]

F=de.iloc[[14360,14720,15100,15560,15920,16220,16560,16920,17280,17600,17920,18220,18500,18780,19020,19240], 209].values
Fpzu=F-de.iloc[[14110], 209].values
Fp = pd.DataFrame(Fpzu)

x1 = np.array([500,4000,7500])
y = np.array(Fp.iloc[[0,  7,14], 0].values)


def func(X, a, b): 
    y = a * X + b 
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]




Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]


F=de.iloc[[14560,14900,15340,15740,16060,16400,16740,17100,17440,17760,18060,18360,18640,18900,19120,19340], 209].values
Fpzd=F-de.iloc[[14110], 209].values
Fp = pd.DataFrame(Fpzd)

x1 = np.array([500,4000,7500])
y = np.array(Fp.iloc[[0,  7,14], 0].values)

def func(X, a, b): 
    y = a * X + b  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]




data_points = [(500, Fpx[0]), (1000, Fpx[1]), (1500, Fpx[2]), (2000, Fpx[3]), (2500, Fpx[4]), (3000, Fpx[5]), (3500, Fpx[6]), (4000, Fpx[7]), (4500, Fpx[8]), (5000, Fpx[9]), (5500, Fpx[10]), (6000, Fpx[11]), (6500, Fpx[12]), (7000, Fpx[13]), (7500, Fpx[14]), (8000, Fpx[15])]

data_points1 = [(500, Fpy[0]), (1000, Fpy[1]), (1500, Fpy[2]), (2000, Fpy[3]), (2500, Fpy[4]), (3000, Fpy[5]), (3500, Fpy[6]), (4000, Fpy[7]), (4500, Fpy[8]), (5000, Fpy[9]), (5500, Fpy[10]), (6000, Fpy[11]), (6500, Fpy[12]), (7000, Fpy[13]), (7500, Fpy[14]), (8000, Fpy[15])]

data_points2 = [(500, Fpzu[0]), (1000, Fpzu[1]), (1500, Fpzu[2]), (2000, Fpzu[3]), (2500, Fpzu[4]), (3000, Fpzu[5]), (3500, Fpzu[6]), (4000,Fpzu[7]), (4500, Fpzu[8]), (5000, Fpzu[9]), (5500, Fpzu[10]), (6000, Fpzu[11]), (6500, Fpzu[12]), (7000, Fpzu[13]), (7500, Fpzu[14]), (8000,Fpzu[15])]

data_points3 = [(500, Fpzd[0]), (1000, Fpzd[1]), (1500, Fpzd[2]), (2000, Fpzd[3]), (2500, Fpzd[4]), (3000, Fpzd[5]), (3500, Fpzd[6]), (4000, Fpzd[7]), (4500, Fpzd[8]), (5000, Fpzd[9]), (5500, Fpzd[10]), (6000, Fpzd[11]), (6500, Fpzd[12]), (7000, Fpzd[13]), (7500, Fpzd[14]), (8000, Fpzd[15])]


def equation(x, m, c):
    return m * x + c

def equation1(x1, m1, c1):
    return m1 * x1 + c1

def equation2(x2, m2, c2):
    return m2 * x2 + c2

def equation3(x3, m3, c3):
    return m3 * x3 + c3


m = 0.012613814977748605
c = 12.258709471242335

m1 = 0.013202696383070034
c1 = -3.9795128407533644

m2 = 0.06957140703824456
c2 = 29.497827243193125

m3 = -0.03428729879961523
c3 = -12.329198659097703



x_data = [point[0] for point in data_points]
y_data = [point[1] for point in data_points]

x_data1 = [point[0] for point in data_points1]
y_data1 = [point[1] for point in data_points1]

x_data2 = [point[0] for point in data_points2]
y_data2 = [point[1] for point in data_points2]

x_data3 = [point[0] for point in data_points3]
y_data3 = [point[1] for point in data_points3]


plt.figure(figsize=(8, 5), dpi=160, linewidth=2.25)  


plt.scatter(x_data, y_data, color='red', s=150)
plt.scatter(x_data1, y_data1, color='orange', s=150)
plt.scatter(x_data2, y_data2, color='green', s=150)
plt.scatter(x_data3, y_data3, color='purple', s=150)


x_values = list(range(min(x_data), max(x_data) + 1))
y_values = [equation(x, m, c) for x in x_values]
plt.plot(x_values, y_values,  color='red', linewidth=3)

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation(x1, m1, c1) for x1 in x_values]
plt.plot(x_values, y_values,  color='orange', linewidth=3)

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation(x2, m2, c2) for x2 in x_values]
plt.plot(x_values, y_values,  color='green', linewidth=3)

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation(x3, m3, c3) for x3 in x_values]
plt.plot(x_values, y_values, color='purple', linewidth=3)



plt.xticks(fontsize=28, fontname='Arial')
plt.yticks(fontsize=28, fontname='Arial')


plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)


plt.xlabel('Feed [mm/min]', fontsize=33, fontname='Arial')
plt.ylabel('Power [W]', fontsize=33, fontname='Arial')


custom_x_ticks = [0, 2000, 4000, 6000, 8000]
plt.xticks(custom_x_ticks)


custom_y_ticks = [-350, 0, 350, 700]
plt.yticks(custom_y_ticks)

plt.tick_params(axis='both', direction='in', width=2, pad=6,length=6)


plt.legend(loc='upper left', frameon=False, fontsize=20)


plt.savefig('Feed_power.svg', format='svg', transparent=True, bbox_inches='tight')

plt.show()


print('P_x=12.26+0.013fx  feed speed Range500-8000')
print('P_y=-3.98+0.013fy  feed speed Range500-8000')
print('P_zu=29.50+0.069fzu  feed speed Range500-8000')
print('P_zd=-12.33-0.034fzd  feed speed Range500-8000')

# %% cutting power model


# AC=de.iloc[[21550,21950,23080,23870,24840,26270,27760,28750,30000], 209].values
# C=de.iloc[[21000,22500,23400,24300,25400,27000,28300,29400,31000], 209].values

AC=de.iloc[[21550,23080,23870,24840,27760,28750,30000], 209].values
C=de.iloc[[21000,23400,24300,25400,28300,29400,31000], 209].values

RC=C-AC

RCD = pd.DataFrame(RC)



x1 = np.array([1592, 382, 0.5,6])
# x2 = np.array([2070, 745, 0.5,13])
x3 = np.array([2548, 1223, 0.5,20])
x4 = np.array([2070, 994, 1, 6])
x5 = np.array([2548, 612, 1, 13])
# x6 = np.array([1592, 573, 1, 20])
x7 = np.array([2548, 917, 1.5, 6])
x8 = np.array([1592, 764, 1.5, 13])
x9 = np.array([2070, 497, 1.5, 20])


y = RC


def func(x, q, d, e, s, f ): 
    y = q*(x[:,0]**d)*(x[:,1]**e)*(x[:,2]**s)*(x[:,3]**f) 
    return y


X = np.vstack((x1,  x3, x4, x5,  x7, x8, x9))


initial_guess = (1.0, 1.0, 1.0, 1.0,1.0)


popt, pcov = curve_fit(func, X, y, p0=initial_guess)

q_fit, d_fit, e_fit, s_fit, f_fit = popt



import numpy as np


matrix = np.array([
    [1592, 382, 0.5,6],
    # [2070, 745, 0.5,13],
    [2548, 1223, 0.5,20],
    [2070, 994, 1, 6],
    [2548, 612, 1, 13],
    # [1592, 573, 1, 20],
    [2548, 917, 1.5, 6],
    [1592, 764, 1.5, 13],
    [2070, 497, 1.5, 20]
])


def equation(row, q, w, e, r, t):
    return q * (row[0] ** w) * (row[1] ** e) * (row[2] ** r) * (row[3] ** t)



q = 0.037
w = 0.222
e = 0.759
r = 0.9
t = 1.109

result_matrix = np.apply_along_axis(equation, axis=1, arr=matrix, q=q, w=w, e=e, r=r, t=t)

print('P_cutting=0.037(n**0.222)(vf**0.759)(ap**0.9)(ae**1.109)')

y_reshaped = y.reshape(-1, 1)
result = np.concatenate((matrix, y_reshaped), axis=1)

result = pd.DataFrame(result, columns=['Spindel speed', 'Feed rate', 'Depth of cut','Width of cut','cutting power'])
print(result)