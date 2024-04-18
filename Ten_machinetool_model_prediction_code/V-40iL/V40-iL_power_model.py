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

df=pd.read_table('V40-iL_modeldata.txt', sep=',')

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

de = de.drop(range(210))
de.reset_index(drop=True, inplace=True)


plt.plot(de["sum"])



# %% standby

base=680

print('P_stadby=680 [W]')

#%% spindle
Sparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]

S=de.iloc[[250,500,850,1160,1450,1750,2050,2350,2650,2950,3250,3550,3875,4175,4450,4775,5075,5375,5675,5975], 209].values
Sp=S-base


x_data = Sparam
y_data = Sp
Spd = pd.DataFrame(Sp)


x1 = np.array([500, 1000, 1500])
y = np.array(Spd.iloc[[0, 1, 2 ], 0].values)


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



data_points = [(500, Sp[0]), (1000, Sp[1]), (1500, Sp[2]) ]
data_points1 = [(1500, Sp[2]), (2000, Sp[3]), (2500, Sp[4]),(3000, Sp[5]), (3500, Sp[6]) ,(4000, Sp[7]) ,(4500, Sp[8])]
data_points2 = [(4500, Sp[8]) , (5000, Sp[9]),(5500, Sp[10]),(6000, Sp[11]), (6500, Sp[12]), (7000, Sp[13]) , (7500, Sp[14]), (8000, Sp[15]), (8500, Sp[16]), (9000, Sp[17]), (9500, Sp[18]), (10000, Sp[19])]



def equation(x, m, c):
    return m * x + c

def equation2(x1, m1, c1,v1):
    return m1 * x1**2 + c1*x1+v1

def equation1(x2, m2, c2,v2):
    return m2 * x2**2 + c2*x2+v2


m = 0.138
c = 46.58

m1 = 1.49*10**-5
c1 = -0.079
v1 = 339.09

m2 = 2.03*10**-5
c2 = -0.17
v2 = 821


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
y_values = [equation2(x1, m1, c1,v1) for x1 in x_values]
plt.plot(x_values, y_values,color='orange')

x_values = list(range(min(x_data2), max(x_data2) + 1))
y_values = [equation1(x2, m2, c2, v2) for x2 in x_values]
plt.plot(x_values, y_values, color='green')



plt.ylim(0, 1300)

custom_x_ticks = [0, 2500, 5000, 7500, 10000]
plt.xticks(custom_x_ticks)


plt.title('V-40iL_spindle', fontsize=33, fontname='Calibri')


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

print('P_spindle=46.58+0.138n  Spindle speed Range500-1500')
print('P_spindle=339.09-0.08n+(1.49*10**-5)n**2  Spindle speed Range1500-4500')
print('P_spindle=821-0.17n+(2.03*10**-5)n**2 Spindle speed Range4500-10000')

# %% feed

Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]

F=de.iloc[[7000,7500,8000,8200,8600,8900,9200,9500,9700,9900,10100,9900,10100,10300,10500,10700], 209].values
Fpx=F-de.iloc[[6450], 209].values


Fp = pd.DataFrame(Fpx)

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
F=de.iloc[[11400,11800,11850,12200,12700,12900,13300,13700,14000,14250,14550,14750,14950,15150,15350,15550], 209].values
Fpy=F-de.iloc[[10900], 209].values


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
Fparam=[1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
F=de.iloc[[17480,17860,18320,18680,18980,19340,19720,20080,20420,20760,21080,21380,21650,21920,22160], 209].values
Fpzu=F-de.iloc[[15800], 209].values
Fp = pd.DataFrame(Fpzu)


x1 = np.array([1000,4000,7500])
y = np.array(Fp.iloc[[0,  6, 13], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]


Fparam=[1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
F=de.iloc[[17650,18100,18500,18825,19150,19500,19900,20240,20600,20900,21200,21500,21780,22020,22260], 209].values
Fpzd=F-de.iloc[[15800], 209].values
Fp = pd.DataFrame(Fpzd)

x1 = np.array([1000,4000,7500])
y = np.array(Fp.iloc[[0, 6, 13], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y

from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]




data_points = [(500, Fpx[0]), (1000, Fpx[1]), (1500, Fpx[2]), (2000, Fpx[3]), (2500, Fpx[4]), (3000, Fpx[5]), (3500, Fpx[6]), (4000, Fpx[7]), (4500, Fpx[8]), (5000, Fpx[9]), (5500, Fpx[10]), (6000, Fpx[11]), (6500, Fpx[12]), (7000, Fpx[13]), (7500, Fpx[14]), (8000, Fpx[15])]

data_points1 = [(500, Fpy[0]), (1000, Fpy[1]), (1500, Fpy[2]), (2000, Fpy[3]), (2500, Fpy[4]), (3000, Fpy[5]), (3500, Fpy[6]), (4000, Fpy[7]), (4500, Fpy[8]), (5000, Fpy[9]), (5500, Fpy[10]), (6000, Fpy[11]), (6500, Fpy[12]), (7000, Fpy[13]), (7500, Fpy[14]), (8000, Fpy[15])]

data_points2 = [ (1000, Fpzu[0]), (1500, Fpzu[1]), (2000, Fpzu[2]), (2500, Fpzu[3]), (3000, Fpzu[4]), (3500, Fpzu[5]), (4000,Fpzu[6]), (4500, Fpzu[7]), (5000, Fpzu[8]), (5500, Fpzu[9]), (6000, Fpzu[10]), (6500, Fpzu[11]), (7000, Fpzu[12]), (7500, Fpzu[13]), (8000,Fpzu[14])]

data_points3 = [ (1000, Fpzd[0]), (1500, Fpzd[1]), (2000, Fpzd[2]), (2500, Fpzd[3]), (3000, Fpzd[4]), (3500, Fpzd[5]), (4000, Fpzd[6]), (4500, Fpzd[7]), (5000, Fpzd[8]), (5500, Fpzd[9]), (6000, Fpzd[10]), (6500, Fpzd[11]), (7000, Fpzd[12]), (7500, Fpzd[13]), (8000, Fpzd[14])]


def equation(x, m, c):
    return m * x + c

def equation1(x1, m1, c1):
    return m1 * x1 + c1

def equation2(x2, m2, c2):
    return m2 * x2 + c2

def equation3(x3, m3, c3):
    return m3 * x3 + c3


m = 0.022
c = -2.44

m1 = 0.025
c1 = -4.00

m2 = 0.115
c2 = 21.68

m3 = -0.04
c3 = -55.14

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
plt.title('V-40iL_feed', fontsize=33, fontname='Calibri')

plt.savefig('Feed_power.svg', format='svg', transparent=True, bbox_inches='tight')

plt.show()


print('P_x=-2.44+0.022fx  feed speed Range500-8000')
print('P_y=-4+0.025fy  feed speed Range500-8000')
print('P_zu=21.68+0.12fzu  feed speed Range500-8000')
print('P_zd=-55.14-0.04fzd  feed speed Range500-8000')

# %% cutting

AC=de.iloc[[22750,24900,26080,26900,27900,29400,30950,32000,33300], 209].values
C=de.iloc[[23750,25250,26400,27220,28400,30200,31300,32400,34100], 209].values
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

# y = np.array([65,251,640,230,446,546,335,597,786])
y = np.array([65,640,230,446,335,597,786])


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

q = 0.024
w = 0.353
e = 0.657
r = 0.795
t = 1.107


result_matrix = np.apply_along_axis(equation, axis=1, arr=matrix, q=q, w=w, e=e, r=r, t=t)


print('P_cutting=0.024(n**0.353)(vf**0.657)(ap**0.795)(ae**1.107)')

y_reshaped = y.reshape(-1, 1)
result = np.concatenate((matrix, y_reshaped), axis=1)

result = pd.DataFrame(result, columns=['Spindel speed', 'Feed rate', 'Depth of cut','Width of cut','cutting power'])
print(result)

