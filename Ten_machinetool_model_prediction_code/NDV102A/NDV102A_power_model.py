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

df=pd.read_table('NDV102A_modeldata.txt', sep=',')
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



de = de.drop(range(790))
de.reset_index(drop=True, inplace=True)
plt.plot(de["sum"])

# %% standby

base=625
print('P_stadby=625 [W]')

#%% spindle
Sparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]

S=de.iloc[[200,500,800,1100,1400,1760,2000,2300,2600,2900,3200,3500,3800,4100,4400,4700,5000,5270,5620,5900], 208].values
Sp=S-base


x_data = Sparam
y_data = Sp
Spd = pd.DataFrame(Sp)




x1 = np.array([500,  1500, 2000, 2500 ])
y = np.array(Spd.iloc[[0,  2, 3, 4], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]


x1 = np.array([2500,3000,3500])
y = np.array(Spd.iloc[[4,5,6], 0].values)


def func(X, a, b,C): 
    y = a * X**2 + b*X+C  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]
c_fit = popt[2]



x1 = np.array([3500,5500,7500,9500,10000])
y = np.array(Spd.iloc[[6,10,14,18,19], 0].values)

def func(X, a, b,C,): 
    y = a*X**2+b*X+C  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]
c_fit = popt[2]



data_points = [(500, Sp[0]), (1000, Sp[1]), (1500, Sp[2]), (2000, Sp[3]), (2500, Sp[4]) ]
data_points1 = [(2500, Sp[4]),(3000, Sp[5]), (3500, Sp[6])  ]
data_points2 = [(3500, Sp[6]),(4000, Sp[8]),(4500, Sp[9]),(5000, Sp[10]),(5500, Sp[10]),(6000, Sp[11]), (6500, Sp[12]), (7000, Sp[13]) , (7500, Sp[14]), (8000, Sp[15]), (8500, Sp[16]), (9000, Sp[17]), (9500, Sp[18]), (10000, Sp[19])]


def equation(x, m, c):
    return m * x + c

def equation1(x1, m1, c1,v1):
    return m1 * x1**2 + c1*x1+v1

def equation2(x2, m2, c2,v2):
    return m2 * x2**2 + c2*x2+v2


m = 0.08
c = 108.24

m1 = 0.00013
c1 =-0.738
v1 =1342.59

m2 =1.37*10**-5
c2 = -0.12
v2=606.35



plt.figure(figsize=(16/2.54, 10/2.54), dpi=160, linewidth=2.25) 

x_data = [point[0] for point in data_points]
y_data = [point[1] for point in data_points]

x_data1 = [point[0] for point in data_points1]
y_data1 = [point[1] for point in data_points1]

x_data2 = [point[0] for point in data_points2]
y_data2 = [point[1] for point in data_points2]




plt.scatter(x_data, y_data,   color='red',s=40)
plt.scatter(x_data1, y_data1, color='orange',s=40)
plt.scatter(x_data2, y_data2,  color='green',s=40)

x_values = list(range(min(x_data), max(x_data) + 1))
y_values = [equation(x, m, c) for x in x_values]
plt.plot(x_values, y_values,  color='red',linewidth=3)

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation1(x1, m1, c1, v1) for x1 in x_values]
plt.plot(x_values, y_values, color='orange',linewidth=3)

x_values = list(range(min(x_data2), max(x_data2) + 1))
y_values = [equation2(x2, m2, c2,v2) for x2 in x_values]
plt.plot(x_values, y_values, color='green',linewidth=3)


plt.ylim(0, 1300)


plt.title('NDV102A', fontsize=33, fontname='Calibri')


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


print('P_spindle=108.24+0.08n  Spindle speed Range500-2500')
print('P_spindle=1342.59-0.74n+0.00013n**2  Spindle speed Range2500-3500')
print('P_spindle=606.35-0.12n+(1.37*10**-5)n**2  Spindle speed Range3500-10000')
# %% feed

# Fx
Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]

F=de.iloc[[6500,6800,7000,7300,7650,7850,8100,8300,8600,8800,9000,9250,9450,9650,9850,10050], 208].values
Fpx=F-(de.iloc[[6260], 208].values-59)
Fp = pd.DataFrame(Fpx)
Fp.iloc[:3]=Fp.iloc[:3]-50

x1 = np.array([500,4000,7500])
y = np.array(Fp.iloc[[0, 7,14], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]




# Fy
Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
F=de.iloc[[10600,10900,11140,11360,11640,11900,12150,12350,12575,12800,13050,13225,13450,13650,13825,14025], 208].values
Fpy=F-de.iloc[[10300], 208].values
Fp= pd.DataFrame(Fpy)
Fp.iloc[-11:]=Fp.iloc[-11:]-50

x1 = np.array([500,4000,7500])
y = np.array(Fp.iloc[[0,  7,14], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]




# Fzu
Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]


F=de.iloc[[14500,14850,15250,15700,16080,16400,16730,17100,17465,17800,18125,18445,18745,19020,19280,19530], 208].values
Fpzu=F-de.iloc[[14300], 208].values
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




# Fzd


Fparam=[500,1000,1500,2000,2500,3000,3500]


F=de.iloc[[14700,15050,15525,15900,16200,16550,16900], 208].values

averge= de.iloc[17600:17700,208].mean()-de.iloc[[14300], 208].values


Fpzd=F-de.iloc[[14300], 208].values
Fp = pd.DataFrame(Fpzd)

x1 = np.array([500,2000,3500])
y = np.array(Fp.iloc[[0, 3,  6], 0].values)


def func(X, a, b): 
    y = a * X + b  
    return y


from scipy.optimize import curve_fit


popt, pcov = curve_fit(func, x1, y)


a_fit = popt[0]
b_fit = popt[1]



data_points = [(500, Fpx[0]-50), (1000, Fpx[1]-50), (1500, Fpx[2]-50), (2000, Fpx[3]), (2500, Fpx[4]), (3000, Fpx[5]), (3500, Fpx[6]), (4000, Fpx[7]), (4500, Fpx[8]), (5000, Fpx[9]), (5500, Fpx[10]), (6000, Fpx[11]), (6500, Fpx[12]), (7000, Fpx[13]), (7500, Fpx[14]), (8000, Fpx[15])]

data_points1 = [(500, Fpy[0]), (1000, Fpy[1]), (1500, Fpy[2]), (2000, Fpy[3]), (2500, Fpy[4]), (3000, Fpy[5]-50), (3500, Fpy[6]-50), (4000, Fpy[7]-50), (4500, Fpy[8]-50), (5000, Fpy[9]-50), (5500, Fpy[10]-50), (6000, Fpy[11]-50), (6500, Fpy[12]-50), (7000, Fpy[13]-50), (7500, Fpy[14]-50), (8000, Fpy[15]-50)]

data_points2 = [(500, Fpzu[0]), (1000, Fpzu[1]), (1500, Fpzu[2]), (2000, Fpzu[3]), (2500, Fpzu[4]), (3000, Fpzu[5]), (3500, Fpzu[6]), (4000,Fpzu[7]), (4500, Fpzu[8]), (5000, Fpzu[9]), (5500, Fpzu[10]), (6000, Fpzu[11]), (6500, Fpzu[12]), (7000, Fpzu[13]), (7500, Fpzu[14]), (8000,Fpzu[15])]

data_points3 = [(500, Fpzd[0]), (1000, Fpzd[1]), (1500, Fpzd[2]), (2000, Fpzd[3]), (2500, Fpzd[4]), (3000, Fpzd[5]), (3500, Fpzd[6]), (4000, -310), (4500, -340), (5000, -375), (5500, -410), (6000, -450), (6500, -480), (7000, -520), (7500, -550), (8000, -590)]


def equation(x, m, c):
    return m * x + c

def equation1(x1, m1, c1):
    return m1 * x1 + c1

def equation2(x2, m2, c2):
    return m2 * x2 + c2

def equation3(x3, m3, c3):
    return m3 * x3 + c3


m = 0.048
c = 14.40

m1 = 0.046
c1 = -10.34

m2 = 0.172
c2 = 3.95

m3 = -0.067
c3 = -26.22



x_data = [point[0] for point in data_points]
y_data = [point[1] for point in data_points]

x_data1 = [point[0] for point in data_points1]
y_data1 = [point[1] for point in data_points1]

x_data2 = [point[0] for point in data_points2]
y_data2 = [point[1] for point in data_points2]

x_data3 = [point[0] for point in data_points3]
y_data3 = [point[1] for point in data_points3]


plt.figure(figsize=(16/2.54, 10/2.54), dpi=80, linewidth=2.25)  


plt.scatter(x_data, y_data, color='red')
plt.scatter(x_data1, y_data1, color='orange')
plt.scatter(x_data2, y_data2, color='green')
plt.scatter(x_data3, y_data3, color='purple')


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

print('P_x=14.40+0.048fx  feed speed Range500-8000')
print('P_y=-10.34+0.046fy  feed speed Range500-8000')
print('P_zu=3.95+0.17fzu  feed speed Range500-8000')
print('P_zd=-26.22-0.067fzd  feed speed Range500-8000')
# %% cutting





# AC=de.iloc[[20250,22500,23750,24660,25760,27260,28860,29980,31340], 208].values
# C=de.iloc[[21100,22900,24040,25000,26300,27750,29225,30500,32150], 208].values

AC=de.iloc[[20250,23750,24660,25760,28860,29980,31340], 208].values
C=de.iloc[[21100,24040,25000,26300,29225,30500,32150], 208].values

RC=C-AC
RCD = pd.DataFrame(RC)

y = RC
x1 = np.array([1592, 382, 0.5,6])
# x2 = np.array([2070, 745, 0.5,13])
x3 = np.array([2548, 1223, 0.5,20])
x4 = np.array([2070, 994, 1, 6])
x5 = np.array([2548, 612, 1, 13])
# x6 = np.array([1592, 573, 1, 20])
x7 = np.array([2548, 917, 1.5, 6])
x8 = np.array([1592, 764, 1.5, 13])
x9 = np.array([2070, 497, 1.5, 20])



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

q = 0.050
w = -0.04
e = 0.963
r = 0.734
t = 1.29


result_matrix = np.apply_along_axis(equation, axis=1, arr=matrix, q=q, w=w, e=e, r=r, t=t)


print('P_cutting=0.050(n**-0.04)(vf**0.963)(ap**0.734)(ae**1.29)')

y_reshaped = y.reshape(-1, 1)
result = np.concatenate((matrix, y_reshaped), axis=1)

result = pd.DataFrame(result, columns=['Spindel speed', 'Feed rate', 'Depth of cut','Width of cut','cutting power'])
print(result)