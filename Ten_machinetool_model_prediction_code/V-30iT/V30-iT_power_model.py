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


HZ=60
samplerate =12500
sampleperiod =1/samplerate
T=1/HZ

df=pd.read_table('V30-iT_modeldata.txt', sep=',')

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


de = de.drop(range(545))
de.reset_index(drop=True, inplace=True)
plt.plot(de["sum"])

# %% Standby

base=462
print('P_stadby=462 [W]')
#%% spindle

Sparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]
S=de.iloc[[200,500,800,1100,1400,1700,2000,2300,2600,2900,3200,3500,3800,4100,4400,4700,5000,5300,5600,5900], 209].values
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



data_points = [(500, Sp[0]), (1000, Sp[1]),(1500, Sp[2]) ]
data_points1 = [(1500, Sp[2]), (2000, Sp[3]), (2500, Sp[4]), (3000, Sp[5]), (3500, Sp[6]) ,(4000, Sp[7]),(4500, Sp[8]) ]
data_points2 = [(4500, Sp[8]),(5000, Sp[9]),(5500, Sp[10]),(6000, Sp[11]), (6500, Sp[12]), (7000, Sp[13]) , (7500, Sp[14]), (8000, Sp[15]), (8500, Sp[16]), (9000, Sp[17]), (9500, Sp[18]), (10000, Sp[19])]


def equation(x, m, c):
    return m * x + c

def equation1(x1, m1, c1,v1):
    return m1 * x1**2 + c1*x1+v1

def equation2(x2, m2, c2,v2):
    return m2 * x2**2 + c2*x2+v2


m = 0.14
c = 82.40

m1 = 1.73*10**-5
c1 = -0.05
v1 = 328.38

m2 = 1.023*10**-5
c2 = -0.035
v2 = 573.03


plt.figure(figsize=(16/2.54, 10/2.54), dpi=80, linewidth=2.25)  


x_data = [point[0] for point in data_points]
y_data = [point[1] for point in data_points]

x_data1 = [point[0] for point in data_points1]
y_data1 = [point[1] for point in data_points1]

x_data2 = [point[0] for point in data_points2]
y_data2 = [point[1] for point in data_points2]


plt.scatter(x_data, y_data,   color='red')
plt.scatter(x_data1, y_data1, color='orange')
plt.scatter(x_data2, y_data2,  color='green')



x_values = list(range(min(x_data), max(x_data) + 1))
y_values = [equation(x, m, c) for x in x_values]
plt.plot(x_values, y_values,  color='red')

x_values = list(range(min(x_data1), max(x_data1) + 1))
y_values = [equation1(x1, m1, c1, v1) for x1 in x_values]
plt.plot(x_values, y_values, color='orange')

x_values = list(range(min(x_data2), max(x_data2) + 1))
y_values = [equation2(x2, m2, c2,v2) for x2 in x_values]
plt.plot(x_values, y_values, color='green')



plt.ylim(0, 1300)


plt.title('V-30iT', fontsize=33, fontname='Calibri')


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


print('P_spindle=82.40+0.14n  Spindle speed Range500-1500')
print('P_spindle=328.38-0.05n+(1.73*10**-5)n**2  Spindle speed Range1500-4500')
print('P_spindle=573.03-0.035n+(1.02*10**-5)n**2  Spindle speed Range4500-10000')

# %% feed


Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]

F=de.iloc[[6500,6900,7250,7650,8050,8300,8650,9000,9150,9500,9750,10000,10200,10400,10600,10850], 209].values
Fpx=F-de.iloc[[6225], 209].values
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
F=de.iloc[[11400,11600,12050,12400,12800,13100,13400,13750,14150,14350,14600,14850,15100,15300,15500,15700], 209].values
Fpy=F-de.iloc[[11000], 209].values
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


F=de.iloc[[16250,16600,17000,17450,17800,18150,18500,18900,19270,19600,19950,20300,20625,20920,21200,21480], 209].values
Fpzu=F-de.iloc[[16000], 209].values
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



# Fzd
Fparam=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]
F=de.iloc[[16400,16800,17250,17650,17950,18300,18700,19100,19450,19800,20150,20450,20760,21060,21340,21600], 209].values
Fpzd=F-de.iloc[[16000], 209].values
Fp = pd.DataFrame(Fpzd)

x1 = np.array([500,4500,7500])
y = np.array(Fp.iloc[[0, 7,14], 0].values)


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


m = 0.025
c = 22.69

m1 = 0.03
c1 = 2.94

m2 = 0.085
c2 = 22.92

m3 = -0.008
c3 = -20.85


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
plt.legend(loc='upper left')


plt.savefig('feed_power.svg', format='svg', transparent=True)


plt.show()

print('P_x=22.69+0.025fx  feed speed Range500-8000')
print('P_y=2.94+0.03fy  feed speed Range500-8000')
print('P_zu=22.92+0.085fzu  feed speed Range500-8000')
print('P_zd=-20.85-0.008fzd  feed speed Range500-8000')

# %% cutting


# AC=de.iloc[[22200,24500,25820,26750,27900,29500,31100,32220,33700], 209].values
# C=de.iloc[[23200,24900,26100,27050,28600,30100,31600,32700,34700], 209].values

AC=de.iloc[[22200,25820,26750,27900,31100,32220,33700], 209].values
C=de.iloc[[23200,26100,27050,28600,31600,32700,34700], 209].values
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



q = 0.127
w = 0.118
e = 0.704
r = 0.76
t = 1.11

result_matrix = np.apply_along_axis(equation, axis=1, arr=matrix, q=q, w=w, e=e, r=r, t=t)

print('P_cutting=0.128(n**0.118)(vf**0.704)(ap**0.76)(ae**1.11)')


y_reshaped = y.reshape(-1, 1)
result = np.concatenate((matrix, y_reshaped), axis=1)

result = pd.DataFrame(result, columns=['Spindel speed', 'Feed rate', 'Depth of cut','Width of cut','cutting power'])
print(result)
