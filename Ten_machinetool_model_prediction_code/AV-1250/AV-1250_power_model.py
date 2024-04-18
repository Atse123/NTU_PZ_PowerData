# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 17:55:45 2023

@author: Tse
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




HZ=60
samplerate =12500
sampleperiod =1/samplerate
T=1/HZ

df=pd.read_table('AV-1250_modeldata.txt', sep=',')
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

de = de.drop(range(1322))
de.reset_index(drop=True, inplace=True)
plt.plot(de["sum"])


# %% standby
base=887

print('P_stadby=887 [W]')

# %% spindle
Sparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]

S=de.iloc[[200,500,800,1100,1400,1700,2000,2300,2600,2900,3200,3500,3800,4100,4400,4700,5000,5340,5640,5960], 209].values

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


x1 = np.array([1500,2500,3000,4000])
y = np.array(Spd.iloc[[2,4,5,7], 0].values)


def func(X, a, b,C): 
    y = a * X**2 + b*X+C  
    return y

from scipy.optimize import curve_fit

popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]
c_fit = popt[2]



x1 = np.array([4000,6000,8000,10000])
y = np.array(Spd.iloc[[7,11,15,19], 0].values)

def func(X, a, b,C): 
    y = a * X**2 + b*X+C  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]
c_fit = popt[2]




data_points = [(500, Sp[0]), (1000, Sp[1]), (1500, Sp[2]) ]
data_points1 = [(1500, Sp[2]), (2000, Sp[3]), (2500, Sp[4]),(3000, Sp[5]), (3500, Sp[6]) ,(4000, Sp[7]) ]
data_points2 = [ (4000, Sp[7]),(4500, Sp[8]) , (5000, Sp[9]),(5500, Sp[10]),(6000, Sp[11]), (6500, Sp[12]), (7000, Sp[13]) , (7500, Sp[14]), (8000, Sp[15]), (8500, Sp[16]), (9000, Sp[17]), (9500, Sp[18]), (10000, Sp[19])]


def equation(x, m, c):
    return m * x + c


def equation1(x1, m1, c1,v1):
    return m1 * x1**2 + c1*x1+v1

def equation2(x2, m2, c2,v2):
    return m2 * x2**2 + c2*x2+v2

# 方程式的參數
m = 0.105
c = 43.12


m1 = 1.8*10**-5
c1 = -0.09
v1 = 294.23


m2 = 1.96*10**-5
c2 = -0.12
v2 = 381.34


x_data = [point[0] for point in data_points]
y_data = [point[1] for point in data_points]

x_data1 = [point[0] for point in data_points1]
y_data1 = [point[1] for point in data_points1]

x_data2 = [point[0] for point in data_points2]
y_data2 = [point[1] for point in data_points2]

plt.figure(figsize=(16/2.54, 10/2.54), dpi=80, linewidth=2.25)  # 將寬度和高度轉換成英寸，1英寸=2.54公分


plt.scatter(x_data, y_data,   color='red')
plt.scatter(x_data1, y_data1, color='orange')
plt.scatter(x_data2, y_data2,  color='green')




x_values = list(range(min(x_data), max(x_data) + 1))
y_values = [equation(x, m, c) for x in x_values]
plt.plot(x_values, y_values, color='red')

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation1(x1, m1, c1,v1) for x1 in x_values]
plt.plot(x_values, y_values,color='orange')

x_values = list(range(min(x_data2), max(x_data2) + 1))
y_values = [equation2(x2, m2, c2, v2) for x2 in x_values]
plt.plot(x_values, y_values, color='green')


plt.ylim(0, 1300)


plt.title('AV1250', fontsize=33, fontname='Calibri')


plt.xticks(fontsize=28, fontname='Calibri')
plt.yticks(fontsize=28,  fontname='Calibri')


plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)



plt.tick_params(axis='both', direction='in', width=2, pad=6,length=6)

plt.xlabel('Times [s]', fontsize=33, fontname='Calibri')
plt.ylabel('Power [W]', fontsize=33, fontname='Calibri')


plt.legend().set_visible(False)
plt.grid(False)


plt.savefig('spindle_power.svg', format='svg', transparent=True, bbox_inches='tight')


plt.show()

print('P_spindle=43.12+0.105n  Spindle speed Range500-1500')
print('P_spindle=294.23-0.09n+(1.8*10**-5)n**2  Spindle speed Range1500-4000')
print('P_spindle=381.34-0.12n+(1.96*10**-5)n**2  Spindle speed Range4000-10000')

# %% Feed

# Fx 
Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]



F=de.iloc[[6700,6800,7050,7250,7450,7700,7950,8200,8450,8700,8950,9200,9450,9650,9900,10140,10320], 209].values


Fpx=F-de.iloc[[6500], 209].values

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


F=de.iloc[[10800,11200,11450,11700,12050,12350,12550,12800,13050,13300,13500,13750,13950,14200,14400,14600], 209].values

Fpy=F-de.iloc[[10600], 209].values

Fp= pd.DataFrame(Fpy)
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


F=de.iloc[[15100,15500,15900,16350,16750,17050,17550,18200,18900,19275,19625,19975,20325,20650,20950,21225], 209].values

Fpzu=F-de.iloc[[14900], 209].values

Fp = pd.DataFrame(Fpzu)

x1 = np.array([500,4500,6500])
y = np.array(Fp.iloc[[0, 8,12], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y

from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]



Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]


F=de.iloc[[15300,15650,16100,16550,16900,17300,17800,18700,19060,19530,19850,20200,20500,20800,21100,21400], 209].values

Fpzd=F-de.iloc[[14900], 209].values

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

data_points2 = [(500, Fpzu[0]), (1000, Fpzu[1]), (1500, Fpzu[2]), (2000, Fpzu[3]), (2500, Fpzu[4]), (3000, Fpzu[5]), (3500, Fpzu[6]), (4000,506.5), (4500, Fpzu[8]), (5000, Fpzu[9]), (5500, Fpzu[10]), (6000, Fpzu[11]), (6500, Fpzu[12]), (7000, Fpzu[13]), (7500, Fpzu[14]), (8000,Fpzu[15])]

data_points3 = [(500, Fpzd[0]), (1000, Fpzd[1]), (1500, Fpzd[2]), (2000, Fpzd[3]), (2500, Fpzd[4]), (3000, -234.4), (3500, Fpzd[6]), (4000, Fpzd[7]), (4500, Fpzd[8]), (5000, Fpzd[9]), (5500, Fpzd[10]), (6000, Fpzd[11]), (6500, Fpzd[12]), (7000, Fpzd[13]), (7500, Fpzd[14]), (8000, Fpzd[15])]


def equation(x, m, c):
    return m * x + c

def equation1(x1, m1, c1):
    return m1 * x1 + c1

def equation2(x2, m2, c2):
    return m2 * x2 + c2

def equation3(x3, m3, c3):
    return m3 * x3 + c3


m = 0.026
c = -12.73

m1 = 0.031
c1 = -3.23

m2 = 0.128
c2 = 18.37

m3 = -0.071
c3 =-27.49


x_data = [point[0] for point in data_points]
y_data = [point[1] for point in data_points]

x_data1 = [point[0] for point in data_points1]
y_data1 = [point[1] for point in data_points1]

x_data2 = [point[0] for point in data_points2]
y_data2 = [point[1] for point in data_points2]

x_data3 = [point[0] for point in data_points3]
y_data3 = [point[1] for point in data_points3]


plt.figure(figsize=(16/2.54, 10/2.54), dpi=80, linewidth=2.25)  # 將寬度和高度轉換成英寸，1英寸=2.54公分



plt.scatter(x_data, y_data,   color='red')
plt.scatter(x_data1, y_data1, color='orange')
plt.scatter(x_data2, y_data2,  color='green')
plt.scatter(x_data3, y_data3,  color='purple')



x_values = list(range(min(x_data), max(x_data) + 1))
y_values = [equation(x, m, c) for x in x_values]
plt.plot(x_values, y_values, label=f'Px', color='red')

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation(x1, m1, c1) for x1 in x_values]
plt.plot(x_values, y_values, label=f'Py', color='orange')

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation(x2, m2, c2) for x2 in x_values]
plt.plot(x_values, y_values, label=f'Pz+', color='green')

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation(x3, m3, c3) for x3 in x_values]
plt.plot(x_values, y_values, label=f'Pz-', color='purple')


plt.ylim(-700, 1500)


plt.xlabel('')
plt.ylabel('')


plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')


plt.gca().spines['top'].set_linewidth(3)
plt.gca().spines['bottom'].set_linewidth(3)
plt.gca().spines['left'].set_linewidth(3)
plt.gca().spines['right'].set_linewidth(3)


plt.tick_params(axis='both', direction='in', width=3, pad=6)


plt.legend()
plt.grid(False)

plt.legend(loc='upper left')

plt.savefig('feed_power.svg', format='svg', transparent=True)


plt.show()

print('P_x=-12.73+0.026fx  feed speed Range500-8000')
print('P_y=-3.23+0.031fy  feed speed Range500-8000')
print('P_zu=18.37+0.128fzu  feed speed Range500-8000')
print('P_zd=-27.49-0.071fzd  feed speed Range500-8000')

# %% cutting

# AC=de.iloc[[22000,24200,25420,26300,27370,28900,30490,31550,32880], 209].values
# C=de.iloc[[22750,24600,25660,26540,27850,29550,30780,32020,33500], 209].values

AC=de.iloc[[22000,25420,26300,27370,30490,31550,32880], 209].values
C=de.iloc[[22750,25660,26540,27850,30780,32020,33500], 209].values
RC=C-AC


x1 = np.array([1592, 382, 0.5,6])
# x2 = np.array([2070, 745, 0.5,13])
x3 = np.array([2548, 1223, 0.5,20])
x4 = np.array([2070, 994, 1, 6])
x5 = np.array([2548, 612, 1, 13])
# x6 = np.array([1592, 573, 1, 20])
x7 = np.array([2548, 917, 1.5, 6])
x8 = np.array([1592, 764, 1.5, 13])
x9 = np.array([2070, 497, 1.5, 20])


y=RC

def func(x, q,w,e,r,t ): 
    y = q*(x[:,0]**w)*(x[:,1]**e)*(x[:,2]**r)*(x[:,3]**t) 
    return y


X = np.vstack((x1,  x3, x4, x5,  x7, x8, x9))


initial_guess = (1.0, 1.0, 1.0, 1.0,1.0)


popt, pcov = curve_fit(func, X, y, p0=initial_guess)


q_fit, w_fit, e_fit, r_fit, t_fit = popt


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


q = 0.022
w = 0.21
e = 0.83
r = 0.83
t = 1.13


result_matrix = np.apply_along_axis(equation, axis=1, arr=matrix, q=q, w=w, e=e, r=r, t=t)

print('P_cutting=0.022(n**0.21)(vf**0.83)(ap**0.83)(ae**1.13)')
y_reshaped = y.reshape(-1, 1)
result = np.concatenate((matrix, y_reshaped), axis=1)

result = pd.DataFrame(result, columns=['Spindel speed', 'Feed rate', 'Depth of cut','Width of cut','cutting power'])
print(result)



