import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import openpyxl
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cProfile import label
from scipy import signal
import math

data = pd.read_table('pattern_NCcode.txt')
dataframe_NCode = np.array(data)
dataframe_NCode = dataframe_NCode.astype(str)


## Parameters
G28_X = 300. 
G28_Y = 200. 
G28_Z = 300. 
Rapid_Feed = 15000
unti_theta = 15
workpiece_lenght = 145
workpiece_width = 131.5
workpiece_depth = 23
D = 20 # Diameter of tool
R = D/2 # Radias of tool
P_stanby = 887

dataframe_ModiftNCode =  pd.DataFrame({"NC-Code": []})
for i in range(0, len(dataframe_NCode)):
    row_i = dataframe_NCode[i][0]
    flag_X= "X" in row_i; flag_Y= "Y" in row_i; flag_Z= "Z" in row_i; flag_S= "S" in row_i; flag_F= "F" in row_i;
    flag_I= "I" in row_i; flag_J= "J" in row_i; flag_M3= "M3" in row_i;
    if flag_X:
        row_i = row_i.replace("X", " X")
    if flag_Y:
        row_i = row_i.replace("Y", " Y")
    if flag_Z:
        row_i = row_i.replace("Z", " Z")
    if flag_S:
        row_i = row_i.replace("S", " S")
    if flag_F:
        row_i = row_i.replace("F", " F")
    if flag_I:
        row_i = row_i.replace("I", " I")
    if flag_J:
        row_i = row_i.replace("J", " J")
    if flag_M3:
        row_i = row_i.replace("M3", " M3 ")
    else:
        row_i = row_i
    Append_dataframe= pd.DataFrame({"NC-Code" : [ row_i ] })
    dataframe_ModiftNCode= pd.concat([dataframe_ModiftNCode, Append_dataframe])
dataframe_ModiftNCode = dataframe_ModiftNCode.reset_index(drop= True)
dataframe_ModiftNCode = np.array(dataframe_ModiftNCode)
    


## Dataframe
#X
X_dataframe= pd.DataFrame({"X": []})
for i in range(0, len(dataframe_ModiftNCode)):
    row_i= dataframe_ModiftNCode[i][0]
    flag_X= "X" in row_i
    flag_G28 ="G28" in row_i
    
    if flag_X and flag_G28 : 
        X_value= G28_X
    elif flag_X:
        X_value= row_i.split("X")[1].split(" ")[0]
    elif flag_G28:
        X_value= G28_X   
    else:
        X_value= np.nan
        
    Xappend_dataframe= pd.DataFrame({"X" : [ X_value ] })
    X_dataframe= pd.concat([X_dataframe, Xappend_dataframe])

X_dataframe = X_dataframe.reset_index(drop= True)

X_dataframe = X_dataframe.fillna(method= "pad")

X_dataframe = X_dataframe.replace(np.nan ,G28_X)
  

#Y
Y_dataframe= pd.DataFrame({"Y": []})
for i in range(0, len(dataframe_ModiftNCode)):
    row_i= dataframe_ModiftNCode[i][0]
    flag_Y= "Y" in row_i
    flag_G28 ="G28" in row_i
    
    if flag_Y and flag_G28 : 
        Y_value= G28_Y
    elif flag_Y:
        Y_value= row_i.split("Y")[1].split(" ")[0]
    elif flag_G28:
        Y_value= G28_Y   
    else:
        Y_value= np.nan
        
    Yappend_dataframe= pd.DataFrame({"Y" : [ Y_value ] })
    Y_dataframe= pd.concat([Y_dataframe, Yappend_dataframe])
    
Y_dataframe = Y_dataframe.reset_index(drop= True)
Y_dataframe = Y_dataframe.fillna(method= "pad")
Y_dataframe = Y_dataframe.replace(np.nan ,G28_Y)


#Z
Z_dataframe= pd.DataFrame({"Z": []})
for i in range(0, len(dataframe_ModiftNCode)):
    row_i= dataframe_ModiftNCode[i][0]
    flag_Z= "Z" in row_i
    flag_G28 ="G28" in row_i
    
    if flag_Z and flag_G28 : 
        Z_value= G28_Z
    elif flag_Z:
        Z_value= row_i.split("Z")[1].split(" ")[0]
    elif flag_G28:
        Z_value= G28_Z   
    else:
        Z_value= np.nan
        
    Zappend_dataframe= pd.DataFrame({"Z" : [ Z_value ] })
    Z_dataframe= pd.concat([Z_dataframe, Zappend_dataframe])

Z_dataframe = Z_dataframe.reset_index(drop= True)
Z_dataframe = Z_dataframe.fillna(method= "pad")
Z_dataframe = Z_dataframe.replace(np.nan ,G28_Z)


#S
S_dataframe = pd.DataFrame({"S":[]})

for i in range(0,len(dataframe_ModiftNCode)):
    row_i= dataframe_ModiftNCode[i][0]
    flag_S = "S" in row_i
    flag_M30 = "M30" in row_i
    
    if flag_S:
        S_value = row_i.split("S")[1].split(" ")[0]
    elif flag_M30:
        S_value = 0
    else:
        S_value = np.nan
    Sappend_dataframe = pd.DataFrame({"S" : [S_value]})
    S_dataframe = pd.concat([S_dataframe , Sappend_dataframe])

S_dataframe= S_dataframe.reset_index(drop= True)
S_dataframe= S_dataframe.fillna(method= "pad")
S_dataframe = S_dataframe.replace(np.nan , 0)


#F
F_dataframe = pd.DataFrame({"F":[]})

for i in range(0,len(dataframe_ModiftNCode)):
    row_i= dataframe_ModiftNCode[i][0]
    
    flag_F = "F" in row_i
    flag_G0 = "G0" in row_i
    flag_G00 = "G00" in row_i
    
    if flag_F:
        F_value = row_i.split("F")[1].split(" ")[0]
    elif flag_G0:
        F_value = Rapid_Feed
    elif flag_G00:
        F_value = Rapid_Feed        
    else:
        F_value = np.nan
    Fappend_dataframe = pd.DataFrame({"F" : [F_value]})
    F_dataframe = pd.concat([F_dataframe , Fappend_dataframe])

F_dataframe= F_dataframe.reset_index(drop= True)
F_dataframe= F_dataframe.fillna(method= "pad")
F_dataframe = F_dataframe.replace(np.nan , 0)


G03_dataframe = pd.DataFrame({"G03":[]})
for i in range(0,len(dataframe_ModiftNCode)):
    row_i= dataframe_ModiftNCode[i][0]
    
    flag_G03 = "G03" in row_i
    flag_G3 = "G3" in row_i
    flag_i ="I" in row_i
    flag_j= "J" in row_i
    
    if flag_G3 and flag_i:
        G03_value = 1
    elif flag_G3 and flag_j:
        G03_value = 1
    elif flag_G03:
        G03_value = 1
    else:
        G03_value = 0
    G03append_dataframe = pd.DataFrame({"G03" : [G03_value]})
    G03_dataframe = pd.concat([G03_dataframe , G03append_dataframe])
G03_dataframe= G03_dataframe.reset_index(drop= True)

G04_dataframe = pd.DataFrame({"G04":[]})

for i in range(0,len(dataframe_ModiftNCode)):
    row_i= dataframe_ModiftNCode[i][0]
    
    flag_G04 = "G02" in row_i
    flag_G4 = "G2" in row_i
    flag_i ="I" in row_i
    flag_j= "J" in row_i
    
    if flag_G4 and flag_i:
        G04_value = 1
    elif flag_G4 and flag_j:
        G04_value = 1
    elif flag_G04:
        G04_value = 1
    else:
        G04_value = 0
    G04append_dataframe = pd.DataFrame({"G04" : [G04_value]})
    G04_dataframe = pd.concat([G04_dataframe , G04append_dataframe])
G04_dataframe= G04_dataframe.reset_index(drop= True)


ij_dataframe= pd.DataFrame({"i": [] , "j": []})
for i in range(0, len(dataframe_ModiftNCode)):
    row_i= dataframe_ModiftNCode[i][0]
    flag_i= "I" in row_i
    flag_j= "J" in row_i 
    if flag_i and flag_j : 
        i_value= row_i.split("I")[1].split(" ")[0]
        j_value= row_i.split("J")[1].split(" ")[0]
    else:
        i_value= np.nan
        j_value= np.nan
         
    ijappend_dataframe= pd.DataFrame({"i": [i_value], "j": [j_value] })
    ij_dataframe= pd.concat([ij_dataframe, ijappend_dataframe])
    
ij_dataframe = ij_dataframe.reset_index(drop= True)
ij_dataframe = ij_dataframe.replace(np.nan , 0)
ij_dataframe =  ij_dataframe.astype(float)

r_dataframe = pd.DataFrame({"r":[]})
for i in range(0,len(dataframe_ModiftNCode)):
    r_value =np.sqrt(np.add(np.square(ij_dataframe["i"][i]),np.square(ij_dataframe["j"][i])))
    r_append_dataframe = pd.DataFrame({"r" : [r_value]})
    r_dataframe = pd.concat([r_dataframe , r_append_dataframe])
r_dataframe = r_dataframe.reset_index(drop = True)

All_dataframe = pd.concat([X_dataframe , Y_dataframe , Z_dataframe , S_dataframe , F_dataframe , ij_dataframe , r_dataframe, G03_dataframe, G04_dataframe],axis =1)
All_dataframe =  All_dataframe.astype(float)

for i in range(0, len(All_dataframe)):
    G03 = All_dataframe["G03"][i]
    G04 = All_dataframe["G04"][i]
    r = All_dataframe["r"][i]
    if r != 0 and  G03 == 0 and G04 == 0:
        All_dataframe["G03"][i] = All_dataframe["G03"][i- 1]
        All_dataframe["G04"][i] = All_dataframe["G04"][i- 1]
    else:
        pass
        

theta_dataframe= pd.DataFrame({"theta": []})
for i in range(0, len(All_dataframe)):
    i_data= All_dataframe["i"].iloc[i]
    j_data= All_dataframe["j"].iloc[i]
    if i_data > 0 and j_data == 0:
        theta_value = 180
    elif i_data < 0 and j_data == 0:
        theta_value = 0
    elif i_data == 0 and j_data > 0:
        theta_value = 270
    elif i_data == 0 and j_data < 0:
        theta_value = 90
    elif i_data < 0 and j_data != 0:
        theta_value = math.degrees(math.atan((-All_dataframe.loc[i,"i"])/(All_dataframe.loc[i,"j"])))
        if theta_value > 0:
            theta_value = theta_value
        else:
            theta_value = 360 + theta_value
    elif i_data > 0 and j_data != 0:
        theta_value = math.degrees(math.atan((-All_dataframe.loc[i,"i"])/(All_dataframe.loc[i,"j"])))
        if theta_value > 0:
            theta_value = 180 - theta_value
        else:
            theta_value = 180 + theta_value
    else:
        theta_value = 0
    theta_append_dataframe= pd.DataFrame({"theta": [theta_value] })
    theta_dataframe= pd.concat([theta_dataframe, theta_append_dataframe])

theta_dataframe = theta_dataframe.reset_index(drop= True)
theta_dataframe = theta_dataframe.astype(float)

#central point

center_dataframe= All_dataframe.copy()
center_dataframe["cen_X"]= np.zeros(shape = (len(All_dataframe)))
center_dataframe["cen_Y"]= np.zeros(shape = (len(All_dataframe)))

for i in range(0, len(center_dataframe)):
    i_data= center_dataframe["i"].iloc[i]
    j_data= center_dataframe["j"].iloc[i]
    x_data= center_dataframe["X"].iloc[i]
    y_data= center_dataframe["Y"].iloc[i]
    
    if i_data != 0 or j_data!= 0:
        
        center_dataframe["cen_X"].iloc[i]= center_dataframe["X"].iloc[i- 1]+ i_data
        center_dataframe["cen_Y"].iloc[i]= center_dataframe["Y"].iloc[i- 1]+ j_data
    else:
        pass
center_dataframe = center_dataframe.astype(float)


new_ALL_dataframe= pd.DataFrame({"X": [],"Y" : [], "Z": [], "S": [], "F": [], "i": [], "j": [],"r": [], "G03": [], "G04": []})

for i in range(0, len(center_dataframe)):
    if All_dataframe["r"][i]== 0:
        new_ALL_dataframe= pd.concat([new_ALL_dataframe, pd.DataFrame(All_dataframe.iloc[i]).T], axis= 0)
        pass

    elif All_dataframe["G03"][i] == 1:
        j= 1
        G3_1_movement_dataframe= pd.DataFrame({"X": [],"Y" : [], "Z": [], "S": [], 
                                               "F": [], "i": [], "j": [],"r": [],
                                               "G03": [], "G04": []})
        while True:
            Radians= math.radians(theta_dataframe["theta"][i]+ j* unti_theta)
            Radians_unit= math.radians(unti_theta/2)
            G3X_value = center_dataframe["cen_X"][i]+ center_dataframe["r"][i]*round(math.cos(Radians), 4)
            G3Y_value = center_dataframe["cen_Y"][i]+ center_dataframe["r"][i]*round(math.sin(Radians), 4)
            G3_iteration_dataframe= pd.DataFrame({"X": [G3X_value], 
                                                  "Y": [G3Y_value], 
                                                  "Z": [center_dataframe["Z"].iloc[i- 1]], 
                                                  "S": [center_dataframe["S"].iloc[i- 1]], 
                                                  "F": [center_dataframe["F"].iloc[i- 1]], 
                                                  "i": [center_dataframe["i"].iloc[i- 1]],
                                                  "j": [center_dataframe["j"].iloc[i- 1]],
                                                  "r": [center_dataframe["r"].iloc[i- 1]],
                                                  "G03": [center_dataframe["G03"].iloc[i]],
                                                  "G04": [center_dataframe["G04"].iloc[i]]})

            G3_1_movement_dataframe= pd.concat([G3_1_movement_dataframe, G3_iteration_dataframe], axis= 0)
            if G3X_value == center_dataframe["X"][i] or G3Y_value == center_dataframe["Y"][i]:
                break
            if abs(2*center_dataframe["r"][i]*round(math.sin(Radians_unit), 4)) > abs((G3X_value- center_dataframe["X"][i])**2 + (G3Y_value- center_dataframe["Y"][i])**2 )**(1/2):
                G3X_value = center_dataframe["X"][i]
                G3Y_value = center_dataframe["Y"][i]
                G3_iteration_dataframe= pd.DataFrame({"X": [G3X_value], 
                                                      "Y": [G3Y_value], 
                                                      "Z": [center_dataframe["Z"].iloc[i- 1]], 
                                                      "S": [center_dataframe["S"].iloc[i- 1]], 
                                                      "F": [center_dataframe["F"].iloc[i- 1]], 
                                                      "i": [center_dataframe["i"].iloc[i- 1]],
                                                      "j": [center_dataframe["j"].iloc[i- 1]],
                                                      "r": [center_dataframe["r"].iloc[i- 1]],
                                                      "G03": [center_dataframe["G03"].iloc[i]],
                                                      "G04": [center_dataframe["G04"].iloc[i]]})
                G3_1_movement_dataframe= pd.concat([G3_1_movement_dataframe, G3_iteration_dataframe], axis= 0)
                break
            else:
                j+= 1
        new_ALL_dataframe= pd.concat([new_ALL_dataframe, G3_1_movement_dataframe], axis= 0)
    elif All_dataframe["G04"][i] == 1:
        k= 1
        G4_1_movement_dataframe= pd.DataFrame({"X": [],"Y" : [], "Z": [], "S": [], 
                                               "F": [], "i": [], "j": [],"r": [],
                                               "G03": [], "G04": []})
        
        while True:
            Radians= math.radians(theta_dataframe["theta"][i]- k* unti_theta)
            Radians_unit= math.radians(unti_theta/2)
            G4X_value = center_dataframe["cen_X"][i]+ center_dataframe["r"][i]*round(math.cos(Radians), 4)
            G4Y_value = center_dataframe["cen_Y"][i]+ center_dataframe["r"][i]*round(math.sin(Radians), 4)
                        
            G4_iteration_dataframe= pd.DataFrame({"X": [G4X_value], 
                                                  "Y": [G4Y_value], 
                                                  "Z": [center_dataframe["Z"].iloc[i- 1]], 
                                                  "S": [center_dataframe["S"].iloc[i- 1]], 
                                                  "F": [center_dataframe["F"].iloc[i- 1]], 
                                                  "i": [center_dataframe["i"].iloc[i- 1]],
                                                  "j": [center_dataframe["j"].iloc[i- 1]],
                                                  "r": [center_dataframe["r"].iloc[i- 1]],
                                                  "G03": [center_dataframe["G03"].iloc[i]],
                                                  "G04": [center_dataframe["G04"].iloc[i]]})
            
            G4_1_movement_dataframe= pd.concat([G4_1_movement_dataframe, G4_iteration_dataframe], axis= 0)
            
            if round(G4X_value, 3) == center_dataframe["X"][i] or G4Y_value == center_dataframe["Y"][i]:
                break
            if abs(2*center_dataframe["r"][i]*round(math.sin(Radians_unit), 4)) > abs((G4X_value- center_dataframe["X"][i])**2 + (G4Y_value- center_dataframe["Y"][i])**2 )**(1/2):
                G4X_value = center_dataframe["X"][i]
                G4Y_value = center_dataframe["Y"][i]
                G4_iteration_dataframe= pd.DataFrame({"X": [G4X_value], 
                                                      "Y": [G4Y_value], 
                                                      "Z": [center_dataframe["Z"].iloc[i- 1]], 
                                                      "S": [center_dataframe["S"].iloc[i- 1]], 
                                                      "F": [center_dataframe["F"].iloc[i- 1]], 
                                                      "i": [center_dataframe["i"].iloc[i- 1]],
                                                      "j": [center_dataframe["j"].iloc[i- 1]],
                                                      "r": [center_dataframe["r"].iloc[i- 1]],
                                                      "G03": [center_dataframe["G03"].iloc[i]],
                                                      "G04": [center_dataframe["G04"].iloc[i]]})
                G4_1_movement_dataframe= pd.concat([G4_1_movement_dataframe, G4_iteration_dataframe], axis= 0)
                break
            else:
                k+= 1
        new_ALL_dataframe= pd.concat([new_ALL_dataframe, G4_1_movement_dataframe], axis= 0)
#       new_ALL_dataframe= pd.concat([new_ALL_dataframe, G3_1_movement_dataframe.iloc[: -1, :]], axis= 0)            

new_ALL_dataframe= new_ALL_dataframe.reset_index(drop= True)



workpiece_array= np.zeros(shape=(int((2*D+ workpiece_width+ 2*D)* 10), int((2*D+ workpiece_lenght+ 2*D)* 10)))
workpiece_array[20*D: round(10*(2*D + workpiece_width)) , 20*D: round(10*(2*D + workpiece_lenght))] = 1


cutting_path= pd.DataFrame({"X": [], "Y": []})

for i in range(0, len(new_ALL_dataframe)):
    x= new_ALL_dataframe["X"].iloc[i]
    y= new_ALL_dataframe["Y"].iloc[i]
    z= new_ALL_dataframe["Z"].iloc[i]
    if z<0:

        cutting_path= pd.concat([cutting_path, pd.DataFrame({"X": [x], "Y": [y]})], axis= 0)
    else:
        pass
cutting_path= cutting_path.reset_index(drop= True)



coordinate_path = pd.DataFrame({"X": [], "Y": []})
coordinate_path["X"] = 10* (cutting_path["X"] + 2* D)
coordinate_path["Y"] = 10* (-1*cutting_path["Y"] + workpiece_width+ 2* D)



movement_dataframe= pd.DataFrame({"point_1_x": [], "point_1_y": [], "point_2_x": [], "point_2_y": [],"point_3_x": [], 
                        "point_3_y": [],"point_4_x": [], "point_4_y": []})

for i in range(0, len(coordinate_path)-1):
    previous_x= coordinate_path["X"][i]
    new_x= coordinate_path["X"][i+ 1]
    
    previous_y= coordinate_path["Y"][i]
    new_y= coordinate_path["Y"][i+ 1]
    
    if previous_x == new_x :
        if new_y< previous_y:

            movement_i= pd.DataFrame({"point_1_x": [previous_x- 10* R],
                                      "point_1_y": [previous_y+ 10* R],
                                      "point_2_x": [previous_x+ 10* R],
                                      "point_2_y": [previous_y+ 10* R],
                                      "point_3_x": [new_x- 10* R],
                                      "point_3_y": [new_y- 10* R],
                                      "point_4_x": [new_x+ 10* R],
                                      "point_4_y": [new_y- 10* R],})
        else:

            movement_i= pd.DataFrame({"point_1_x": [new_x- 10* R],
                                      "point_1_y": [new_y+ 10* R],
                                      "point_2_x": [new_x+ 10* R],
                                      "point_2_y": [new_y+ 10* R],
                                      "point_3_x": [previous_x- 10* R],
                                      "point_3_y": [previous_y- 10* R],
                                      "point_4_x": [previous_x+ 10* R],
                                      "point_4_y": [previous_y- 10* R],})    
    elif previous_y == new_y :
        if new_x> previous_x:

            movement_i= pd.DataFrame({"point_1_x": [previous_x- 10* R],
                                      "point_1_y": [previous_y+ 10* R],
                                      "point_2_x": [new_x+ 10* R],
                                      "point_2_y": [new_y+ 10* R],
                                      "point_3_x": [previous_x- 10* R],
                                      "point_3_y": [previous_y- 10* R],
                                      "point_4_x": [new_x+ 10* R],
                                      "point_4_y": [new_y- 10* R],})
        else:

            movement_i= pd.DataFrame({"point_1_x": [new_x- 10* R],
                                      "point_1_y": [new_y+ 10* R],
                                      "point_2_x": [previous_x+ 10* R],
                                      "point_2_y": [previous_y+ 10* R],
                                      "point_3_x": [new_x- 10* R],
                                      "point_3_y": [new_y- 10* R],
                                      "point_4_x": [previous_x+ 10* R],
                                      "point_4_y": [previous_y- 10* R],})
            
    elif (previous_x != new_x) & (previous_y != new_y):
        slope= (new_y- previous_y)/ (new_x- previous_x)
        slope_T= 1/ (slope)* -1
        theta_1= abs(math.degrees(math.atan(slope_T)))+ 45
        theta_2= 45- abs(math.degrees(math.atan(slope_T)))
        

        if (previous_x > new_x) & (previous_y > new_y):

            movement_i= pd.DataFrame({"point_1_x": [previous_x- (2** (1/ 2)* 10* R)* np.cos(theta_1)],
                                      "point_1_y": [previous_y+ (2** (1/ 2)* 10* R)* np.sin(theta_1)],
                                      "point_2_x": [previous_x+ (2** (1/ 2)* 10* R)* np.cos(theta_2)],
                                      "point_2_y": [previous_y- (2** (1/ 2)* 10* R)* np.sin(theta_2)],
                                      "point_3_x": [new_x- (2** (1/ 2)* 10* R)* np.cos(theta_2)],
                                      "point_3_y": [new_y- (2** (1/ 2)* 10* R)* np.sin(theta_2)],
                                      "point_4_x": [new_x+ (2** (1/ 2)* 10* R)* np.cos(theta_1)],
                                      "point_4_y": [new_y- 10* R],
                                     })
            
        elif (previous_x < new_x) & (previous_y > new_y):

            movement_i= pd.DataFrame({"point_1_x": [previous_x- (2** (1/ 2)* 10* R)* np.cos(theta_2)],
                                      "point_1_y": [previous_y+ (2** (1/ 2)* 10* R)* np.sin(theta_2)],
                                      "point_2_x": [previous_x+ (2** (1/ 2)* 10* R)* np.cos(theta_1)],
                                      "point_2_y": [previous_y+ (2** (1/ 2)* 10* R)* np.sin(theta_1)],
                                      "point_3_x": [new_x- (2** (1/ 2)* 10* R)* np.cos(theta_1)],
                                      "point_3_y": [new_y- (2** (1/ 2)* 10* R)* np.sin(theta_1)],
                                      "point_4_x": [new_x+ (2** (1/ 2)* 10* R)* np.cos(theta_2)],
                                      "point_4_y": [new_y- (2** (1/ 2)* 10* R)* np.sin(theta_2)],
                                     })
        
        elif (previous_x > new_x) & (previous_y < new_y):

            movement_i= pd.DataFrame({"point_1_x": [new_x- (2** (1/ 2)* 10* R)* np.cos(theta_2)],
                                      "point_1_y": [new_y+ (2** (1/ 2)* 10* R)* np.sin(theta_2)],
                                      "point_2_x": [new_x+ (2** (1/ 2)* 10* R)* np.cos(theta_1)],
                                      "point_2_y": [new_y+ (2** (1/ 2)* 10* R)* np.sin(theta_1)],
                                      "point_3_x": [previous_x- (2** (1/ 2)* 10* R)* np.cos(theta_1)],
                                      "point_3_y": [previous_y- 10* R],
                                      "point_4_x": [previous_x+ (2** (1/ 2)* 10* R)* np.cos(theta_2)],
                                      "point_4_y": [previous_y+ (2** (1/ 2)* 10* R)* np.sin(theta_2)],
                                     })
            
        else:

            movement_i= pd.DataFrame({"point_1_x": [new_x- (2** (1/ 2)* 10* R)* np.cos(theta_1)],
                                      "point_1_y": [new_y- (2** (1/ 2)* 10* R)* np.sin(theta_1)],
                                      "point_2_x": [new_x+ (2** (1/ 2)* 10* R)* np.cos(theta_2)],
                                      "point_2_y": [new_y- (2** (1/ 2)* 10* R)* np.sin(theta_2)],
                                      "point_3_x": [previous_x- (2** (1/ 2)* 10* R)* np.cos(theta_2)],
                                      "point_3_y": [previous_y- (2** (1/ 2)* 10* R)* np.sin(theta_2)],
                                      "point_4_x": [previous_x+ (2** (1/ 2)* 10* R)* np.cos(theta_1)],
                                      "point_4_y": [new_y- (2** (1/ 2)* 10* R)* np.sin(theta_1)],                
                                     })
            
    movement_dataframe= pd.concat([movement_dataframe, movement_i], axis= 0)

movement_dataframe= movement_dataframe.reset_index(drop= True)
movement= movement_dataframe.round()



results_Step= {}
XY_dataframe= pd.DataFrame({"x1": [], "x2": [], "y1": [], "y2": []})
for i in range(0, len(movement)):
    Step_workpiece= np.zeros(shape=(int((workpiece_width+ 4* D)* 10 ), int((workpiece_lenght+ 4* D)* 10 )))
    x1= int(movement.loc[i, ["point_1_x", "point_2_x", "point_3_x", "point_4_x"]].min())
    x2= int(movement.loc[i, ["point_1_x", "point_2_x", "point_3_x", "point_4_x"]].max())
    y1= int(movement.loc[i, ["point_1_y", "point_2_y", "point_3_y", "point_4_y"]].min())
    y2= int(movement.loc[i, ["point_1_y", "point_2_y", "point_3_y", "point_4_y"]].max())
    XY_append_dataframe = pd.DataFrame({"x1": [x1], "x2": [x2], "y1": [y1], "y2": [y2]})
    XY_dataframe= pd.concat([XY_dataframe, XY_append_dataframe], axis= 0)
    Step_workpiece[y1: y2, x1: x2]= -10
    results_Step["step_"+ str(i)]= Step_workpiece
    
XY_dataframe = XY_dataframe.reset_index(drop = True)

results = {}
results_cutting = {}
Area_dataframe = pd.DataFrame(columns=["Area"])
Extract_Width_Dataframe = pd.DataFrame(columns=["Width"])

for i in range(0, len(movement_dataframe)):
    workpiece_array = workpiece_array + results_Step["step_"+ str(i)]    
    previous_x= coordinate_path["X"][i]
    new_x= coordinate_path["X"][i+ 1]
    previous_y= coordinate_path["Y"][i]
    new_y= coordinate_path["Y"][i+ 1]

    if previous_x == new_x :
            number = np.sum(workpiece_array[int(round((XY_dataframe["y1"][i]+XY_dataframe["y2"][i])/2, 0)),:]== -9)/10
    elif previous_y == new_y :
            number = np.sum(workpiece_array[:, int(round((XY_dataframe["x1"][i]+XY_dataframe["x2"][i])/2, 0))]== -9)/10
    elif (previous_x > new_x) & (previous_y > new_y):
            number = np.sum(workpiece_array[:, int(round(XY_dataframe["x1"][i] + 3 , 0))]== -9)/10
    else:
        number = np.sum(workpiece_array[int(round((XY_dataframe["y1"][i]+XY_dataframe["y2"][i])/2, 0)),:]== -9)/10
    Number_append_dataframe = pd.DataFrame({"Width" : [number]})
    Extract_Width_Dataframe = pd.concat([Extract_Width_Dataframe, Number_append_dataframe], ignore_index=True)
    Area_value =  np.sum(workpiece_array== -9)
    results_cutting["Cutting_"+ str(i)] = workpiece_array
    workpiece_array[workpiece_array != 1] = 0
    Area_append_dataframe = pd.DataFrame({"Area" : [Area_value]})
    Area_dataframe = pd.concat([Area_dataframe, Area_append_dataframe], ignore_index=True)
    results["STEP_CUTTING_"+ str(i)] = workpiece_array

temp_df = pd.DataFrame([[0]], columns=Extract_Width_Dataframe.columns)


Extract_Width_Dataframe = pd.concat([temp_df, Extract_Width_Dataframe], ignore_index=True)


Area_dataframe = Area_dataframe.reset_index(drop=True)
Extract_Width_Dataframe = Extract_Width_Dataframe.reset_index(drop=True)
 

Width_Dataframe= pd.DataFrame({"Width": []})
j = 0

for i in range(0, len(new_ALL_dataframe)):
    x= new_ALL_dataframe["X"].iloc[i]
    y= new_ALL_dataframe["Y"].iloc[i]
    z= new_ALL_dataframe["Z"].iloc[i]

    if z<0:

        Width = Extract_Width_Dataframe["Width"].iloc[j]
        if Width > D:
            Width = D
        j += 1
    else:
        Width = 0
    Width_append_dataframe = pd.DataFrame({"Width" : [Width]})
    Width_Dataframe = pd.concat([Width_Dataframe , Width_append_dataframe])
Width_Dataframe= Width_Dataframe.reset_index(drop= True)




Depth_Dataframe= pd.DataFrame({"Depth": []})
for i in range(0, len(new_ALL_dataframe)):
    depth = 0.5 
    Depth_append_dataframe = pd.DataFrame({"Depth" : [depth]})
    Depth_Dataframe = pd.concat([Depth_Dataframe , Depth_append_dataframe])
Depth_Dataframe = Depth_Dataframe.reset_index(drop = True)

new_ALL_dataframe= pd.concat([new_ALL_dataframe , Width_Dataframe, Depth_Dataframe],axis =1)

# %%

Distance_X =np.diff(new_ALL_dataframe["X"].astype(float),axis=0)
Distance_Y =np.diff(new_ALL_dataframe["Y"].astype(float),axis=0)
Distance_Z =np.diff(new_ALL_dataframe["Z"].astype(float),axis=0)
X = [0]
Y = [0]
Z = [0]


Distance_X = np.r_[X,Distance_X]
Distance_Y = np.r_[Y,Distance_Y]
Distance_Z = np.r_[Z,Distance_Z]

sq_dist_X = np.square(Distance_X)
sq_dist_Y = np.square(Distance_Y)
sq_dist_Z = np.square(Distance_Z)

total_dist_dataframe = pd.DataFrame({"distance":[]})
for i in range(0,len(sq_dist_X)):
    total_value =np.sqrt(np.add(np.add(sq_dist_X[i],sq_dist_Y[i]),sq_dist_Z[i]))
    total_append_dataframe = pd.DataFrame({"distance" : [total_value]})
    total_dist_dataframe = pd.concat([total_dist_dataframe , total_append_dataframe])

total_dist_dataframe = total_dist_dataframe.reset_index(drop = True)
total_dist_dataframe = total_dist_dataframe.values

real_F_dataframe = new_ALL_dataframe["F"].astype(float)/60

with np.errstate(divide = 'ignore' , invalid ='ignore') :
    slope_dataframe = pd.DataFrame({"x":[],"y":[],"z":[]})
    for i in range(0,len(Distance_X)):
        slope_x = Distance_X[i]/total_dist_dataframe[i]
        slope_x[~np.isfinite(slope_x)] = 0
        slope_y = Distance_Y[i]/total_dist_dataframe[i]
        slope_y[~np.isfinite(slope_y)] = 0
        slope_z = Distance_Z[i]/total_dist_dataframe[i]
        slope_z[~np.isfinite(slope_z)] = 0
        slope_append_dataframe = pd.DataFrame({"x" : [abs(slope_x)],"y" : [abs(slope_y)],"z" : [slope_z]})
        slope_dataframe = pd.concat([slope_dataframe , slope_append_dataframe])
slope_dataframe= slope_dataframe.reset_index(drop= True)
    
Fx = slope_dataframe.iloc[:,0]*real_F_dataframe*60
Fy = slope_dataframe.iloc[:,1]*real_F_dataframe*60
Fz = slope_dataframe.iloc[:,2]*real_F_dataframe*60


with np.errstate(divide = 'ignore' , invalid ='ignore') :
    Time_all_dataframe = pd.DataFrame({"T" : []})
    for i in range(0,len(sq_dist_X)):
        T_value = total_dist_dataframe[i]/real_F_dataframe[i]
        T_value[~np.isfinite(T_value)] = 0
        Time_append_dataframe = pd.DataFrame({"T" : T_value})
        Time_all_dataframe = pd.concat([Time_all_dataframe , Time_append_dataframe])

Time_all_dataframe = Time_all_dataframe.reset_index(drop = True)
    


def spindle(s):
    if s== 0.0:
        return 0.0
    elif s <= 2500:
        ps= 43.12+0.105*s
        return ps
    elif  2500 < s <= 5000:
        ps= 294.23-0.09*s+(1.8*10**-5)*s**2
        return ps
    elif   5000 < s:
        ps= (1.96*10**-5)* (s**2)*-0.12*s+381.34
        return ps
    
    
def x_axis(x1):
    if x1== 0.0:
        return 0.0
    else:
        px= 0.026*x1-12.73
        return px

def y_axis(y1):
    if y1== 0.0:
        return 0.0
    else:
        py= 0.031*y1-3.23
        return py

def z_axis(z1):
    if z1 > 0:
        pz= 0.128*z1+18.37
        return pz
    elif z1 < 0:
        pz= -0.071*(-z1)-27.49
        return pz   
    elif z1 == 0.0:
        return 0.0

## (spindle feed depth width)
a = 0.022
b = 0.21
c = 0.83
d = 0.83
e = 1.13

def P_cutting(x1,a,b,c,d,e):
    if x1[0]== 0:
        term_1= 0
    else:
        term_1= a*(x1[0]**b)
    
    if x1[1]== 0:
        term_2= 0
    else:
        term_2= (x1[1]**c)
        
    if x1[2]== 0:
        term_3= 0
    else:
        term_3= (x1[2]**d)
    
    if x1[3]== 0:
        term_4= 0
    else:
        term_4= (x1[3]**e)
    
    Pc= term_1* term_2* term_3* term_4
    return Pc
    

#XY
Fx = np.array(Fx.astype(float))
Fy = np.array(Fy.astype(float))
Fz = np.array(Fz.astype(float))

Power_X = np.zeros_like(Distance_X)
Power_Y = np.zeros_like(Distance_X)
for i in range(0,len(Distance_X)):
    Power_X_i = Fx[i]
    Power_Y_i = Fy[i]
     
    px= x_axis(Power_X_i)
    py= y_axis(Power_Y_i)

    Power_X[i]= px
    Power_Y[i]= py
    
#Power_XY = np.array(Power_XY.astype(float))

#S
S_dataframe = np.array(S_dataframe.astype(float))
Power_S= np.zeros_like(Fz)

for i in range(0,len(Fz)):
    Power_s= new_ALL_dataframe["S"][i]
    
    ps= spindle(Power_s)
    
    Power_S[i]= ps
Power_S = np.array(Power_S)

#Z
Power_Z= np.zeros_like(Fz)

for i in range(0,len(Fz)):
    Power_z= Fz[i]
    
    pz= z_axis(Power_z)
    
    Power_Z[i]= pz

#cutting power
    
Cutting_Dataframe = pd.concat([new_ALL_dataframe["S"] , new_ALL_dataframe["F"], new_ALL_dataframe["Depth"], new_ALL_dataframe["Width"]],axis =1)
Cutting_Dataframe = np.array(Cutting_Dataframe.astype(float))
P_Cutting = pd.DataFrame({"Power":[]})
Power_cutting = np.zeros_like(new_ALL_dataframe["S"])

for i in range(0, len(new_ALL_dataframe)):
    Power = Cutting_Dataframe[i]
    Pc = P_cutting(Power,a,b,c,d,e)
    Power_cutting[i] = Pc


P_feed = Power_X+ Power_Y+ Power_Z

P_cooler=0
P_stanby = np.ones((len(P_feed)) ,dtype=float)*P_stanby
P_total = P_stanby + P_feed + Power_cutting + Power_S+P_cooler

x= np.cumsum(Time_all_dataframe).values.ravel()
y=np.array(pd.DataFrame(P_total)).ravel()


time_values = Time_all_dataframe.iloc[:19].astype(float)
time_sum = np.floor(time_values.sum().values[0])

Energy_comsumption = (((Time_all_dataframe.astype(float).values[:19,0]*y[:19]).sum()))/(1000)


plt.figure(figsize=(8, 5), dpi=160, linewidth=2.25)  

print("prediction_time[s]:", time_sum)
print("prediction_energy[kJ]:", Energy_comsumption)

file_path = 'AV-1250pattern_measuermentpowerdata.xlsx'  
data = pd.read_excel(file_path)


x_data = data["T"]  
y_data = data["P"]  



last_T_value = x_data.iloc[-1]  
print("measurement_time[s]:", last_T_value)

Measurement_Power=sum(y_data)*0.0167
print("measurement_energyconsumption[kJ]:", Measurement_Power/(1000))

plt.plot(x_data, y_data, color='black', label='Measurement') 



plt.step(x, y, linestyle='--', color='darkred',linewidth=2.5, label='Prediction')


custom_y_ticks = [ 0, 1000, 2000,3000,4000,5000]
plt.yticks(custom_y_ticks)

plt.xlim([0, 74])

plt.grid(False)

plt.legend(loc="best")


plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)


plt.xticks(fontsize=28, fontname='Arial')
plt.yticks(fontsize=28, fontname='Arial')


plt.tick_params(axis='both', direction='in', width=2, pad=6,length=6)



plt.xlabel('Time [s]', fontsize=33, fontname='Arial')
plt.ylabel('Power [W]', fontsize=33, fontname='Arial')


plt.legend(loc='upper center', frameon=False, fontsize=28)

plt.title('AV-1250',fontsize=28, fontname='Arial')


plt.show()


plt.savefig('step_plot_transparent.svg', format='svg', transparent=True, bbox_inches='tight')


