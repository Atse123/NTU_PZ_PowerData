## Rapid Energy Consumption Modeling for CNC Based Milling Process
+ This program introduces a novel methodology for swiftly deriving an energy consumption model based on NC code for CNC milling operations. 
+ First, download the Python code for 10 machines, which includes both modeling and prediction programs. Additionally, download the "Full power data" stored on Google Cloud from the URL provided below.
---
## Modeling and predicting energy consumption program.
+ VP-6_power_model.py: This program is designed to create a power model for the VP6 machine tool, which includes models for standby power, spindle power, XYZ axis power, and cutting power.
+ VP-6_power_prediction.py: This program is designed to predict the energy consumption of the VP6 machine tool during machining. It includes cutting parameters extracted from the NC-code, which are then applied to the power model.
+ CNC machine tool power model experiments: Spindle speed (500-10000) interval 500 rpm ,X,Y,Zup and Zdown feed rate (500-8000) interval 500 mm/min ,cutting experiment.


---
## Full power data
+ VP-6_modeldata: Using VP-6_modeldata.txt as an input for VP-6_power_model.py allows for the generation of power models for each component of the machine tool.
+ pattern_NCcode: Using pattern_NCcode.txt as an input for VP-6_power_prediction.py, it predicts the energy consumption of the machining process.
+ VP-6pattern_measuermentpowerdata: Using VP-6_pattern_measurement_power_data.xlsx as an input for VP-6_power_prediction.py, it validates the error in the predicted energy consumption of the machining process.

Full data download: <https://drive.google.com/file/d/1VAVq2c0Jf48yRTvCnKYfFOXgxZGr9nQj/view?usp=sharing>
