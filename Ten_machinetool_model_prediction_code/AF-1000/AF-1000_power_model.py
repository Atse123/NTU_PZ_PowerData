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

df=pd.read_table('AF-1000_modeldata.txt', sep=',')

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


de = de.drop(range(415))
de.reset_index(drop=True, inplace=True)
plt.plot(de["sum"])



# %% standby


base=864

print('P_stadby=864 [W]')

# %%spindle

Sparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]
S=de.iloc[[200,500,800,1160,1350,1700,2000,2300,2600,2900,3260,3570,3820,4100,4360,4700,5000,5300,5580,5880], 209].values
Sp=S-base


x_data = Sparam
y_data = Sp
Spd = pd.DataFrame(Sp)


x1 = np.array([500,  1500, 2000, 2500])
y = np.array(Spd.iloc[[0,  2, 3, 4], 0].values)


def func(X, a, b): 
    y = a * X + b 
    return y


from scipy.optimize import curve_fit

popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]



x1 = np.array([2500,3000,3500,4000])
y = np.array(Spd.iloc[[4,5,6,7], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y

from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]



x1 = np.array([4500,6500,8500,10000])
y = np.array(Spd.iloc[[8,12,16,19], 0].values)

def func(X, a, b,C): 
    y = a * X**2 + b*X+C  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]
c_fit = popt[2]




data_points = [(500, Sp[0]), (1000, Sp[1]), (1500, Sp[2]), (2000, Sp[3]), (2500, Sp[4]) ]
data_points1 = [(2500, Sp[4]),(3000, Sp[5]), (3500, Sp[6]) ,(4000, Sp[7]),(4500, Sp[8])  ]
data_points2 = [(4500, Sp[8]) , (5000, Sp[9]),(5500, Sp[10]),(6000, Sp[11]), (6500, Sp[12]), (7000, Sp[13]) , (7500, Sp[14]), (8000, Sp[15]), (8500, Sp[16]), (9000, Sp[17]), (9500, Sp[18]), (10000, Sp[19])]



def equation(x, m, c):
    return m * x + c

def equation2(x1, m1, c1):
    return m1 * x1 + c1

def equation1(x2, m2, c2,v2):
    return m2 * x2**2 + c2*x2+v2


m = 0.049
c = 43.65

m1 = -0.021
c1 = 209.61

m2 = 1.04*10**-5
c2 = -0.13
v2 = 725



x_data = [point[0] for point in data_points]
y_data = [point[1] for point in data_points]

x_data1 = [point[0] for point in data_points1]
y_data1 = [point[1] for point in data_points1]

x_data2 = [point[0] for point in data_points2]
y_data2 = [point[1] for point in data_points2]


plt.figure(figsize=(16/2.54, 10/2.54), dpi=80, linewidth=2.25)  

plt.scatter(x_data, y_data,   color='red')
plt.scatter(x_data1, y_data1, color='orange')
plt.scatter(x_data2, y_data2,  color='green')



x_values = list(range(min(x_data), max(x_data) + 1))
y_values = [equation(x, m, c) for x in x_values]
plt.plot(x_values, y_values, color='red')

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation2(x1, m1, c1) for x1 in x_values]
plt.plot(x_values, y_values,color='orange')

x_values = list(range(min(x_data2), max(x_data2) + 1))
y_values = [equation1(x2, m2, c2, v2) for x2 in x_values]
plt.plot(x_values, y_values, color='green')



plt.ylim(0, 1300)


plt.title('AF-1000', fontsize=33, fontname='Calibri')


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

print('P_spindle=43.65+0.049n  Spindle speed Range500-2500')
print('P_spindle=209.61-0.02n  Spindle speed Range2500-4500')
print('P_spindle=701.11-0.13n+(1.04*10**-5)n**2  Spindle speed Range4500-10000')

# %% feed


Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
Fpx=([ 27.70854812,  35.57620494,  51.2228961 ,  63.3874022 ,
        77.98244601,  90.02709901, 107.81277868, 121.32592358,
       143.20095839, 153.92259171, 178.93744448, 193.06501887,
       205.08685242, 225.2098445 , 246.73664256, 265.29992553])
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

Fpy=([ 23.58639858,  39.16012817,  57.10880261,  75.77912522,
        84.04475797, 102.17677954, 117.33743659, 137.18839737,
       154.31595945, 172.81003091, 184.21144003, 209.4285746 ,
       226.75727114, 247.74264506, 258.39001103, 281.47065415])
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




Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]

Fpzu=([ 72.20471064, 124.81309896, 178.33006239, 239.68436762,
       293.21554364, 343.91857691, 389.76742518, 455.12403378,
       509.11059296, 574.91702481, 634.65241172, 697.187149  ,
       745.16888528, 812.92192021, 868.51567419, 933.92607352])
Fp = pd.DataFrame(Fpzu)

x1 = np.array([500,4000,7500])
y = np.array(Fp.iloc[[0, 7,14], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]




Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
F=de.iloc[[15750,16100,16600,16950,17250,17600,18000,18320,18650,18960,19300,19600,19850,20100,20350,20560], 209].values
Fpzd=F-de.iloc[[15400], 209].values
Fpzd=([ -42.15047625,  -92.15606251, -122.07877598, -153.68923729,
       -174.59075126, -188.84447435, -186.13807292, -174.17774245,
       -189.96755531, -284.98152176, -305.43052769, -343.1241383 ,
       -390.82448052, -421.29844641, -422.61215513, -474.69684436])
Fp = pd.DataFrame(Fpzd)


x1 = np.array([500,3000,5500])
y = np.array(Fp.iloc[[0,  5,10], 0].values)


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

data_points3 = [(500, Fpzd[0]), (1000, Fpzd[1]), (1500, Fpzd[2]), (2000, Fpzd[3]), (2500, Fpzd[4]), (3000, Fpzd[5]), (3500, -208.4), (4000, -235.4), (4500, -262), (5000, Fpzd[9]), (5500, Fpzd[10]), (6000, Fpzd[11]), (6500, Fpzd[12]), (7000, Fpzd[13]), (7500, Fpzd[14]), (8000, Fpzd[15])]


def equation(x, m, c):
    return m * x + c

def equation1(x1, m1, c1):
    return m1 * x1 + c1

def equation2(x2, m2, c2):
    return m2 * x2 + c2

def equation3(x3, m3, c3):
    return m3 * x3 + c3


m = 0.031
c = 6.76

m1 = 0.033
c1 = 5.55

m2 = 0.114
c2 = 10.25

m3 = -0.053
c3 = -20.84


x_data = [point[0] for point in data_points]
y_data = [point[1] for point in data_points]

x_data1 = [point[0] for point in data_points1]
y_data1 = [point[1] for point in data_points1]

x_data2 = [point[0] for point in data_points2]
y_data2 = [point[1] for point in data_points2]

x_data3 = [point[0] for point in data_points3]
y_data3 = [point[1] for point in data_points3]


plt.figure(figsize=(16/2.54, 10/2.54), dpi=80, linewidth=2.25)  


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


plt.savefig('feed_power.svg', format='svg', transparent=True)


plt.show()

print('P_x=6.76+0.031fx  feed speed Range500-8000')
print('P_y=5.55+0.034fy  feed speed Range500-8000')
print('P_zu=10.25+0.114fzu  feed speed Range500-8000')
print('P_zd=-20.84-0.053fzd  feed speed Range500-8000')

# %% cutting


x1 = np.array([1592, 382, 0.5,6])
# x2 = np.array([2070, 745, 0.5,13])
x3 = np.array([2548, 1223, 0.5,20])
x4 = np.array([2070, 994, 1, 6])
x5 = np.array([2548, 612, 1, 13])
# x6 = np.array([1592, 573, 1, 20])
x7 = np.array([2548, 917, 1.5, 6])
x8 = np.array([1592, 764, 1.5, 13])
x9 = np.array([2070, 497, 1.5, 20])

# y = np.array([56,233,599,251,416,621,354,733,779])
y = np.array([56,599,251,416,354,733,779])


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

q = 0.17
w = -0.066
e = 0.86
r = 0.94
t = 1.09

result_matrix = np.apply_along_axis(equation, axis=1, arr=matrix, q=q, w=w, e=e, r=r, t=t)

print('P_cutting=0.13(n**-0.01)(vf**0.88)(ap**0.96)(ae**1.10)')
y_reshaped = y.reshape(-1, 1)
result = np.concatenate((matrix, y_reshaped), axis=1)

result = pd.DataFrame(result, columns=['Spindel speed', 'Feed rate', 'Depth of cut','Width of cut','cutting power'])
print(result)



