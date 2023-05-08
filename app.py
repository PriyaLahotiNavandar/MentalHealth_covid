from flask import Flask, jsonify,request
import numpy as np
import pickle
app=Flask(__name__)
LinearModel=pickle.load(open("USHouseHoldMentalHealth-Model.pkl","rb"))
ColumnList=pickle.load(open("USHouseHoldMentalHealth.obj","rb"))

@app.route("/")
def status():
    return jsonify({"Message" : "Status Active"})

@app.route("/predict_value", methods = ["POST"]) # api or endpoint
def predict_value():
    data = request.get_json()
    print(data)

    Phase=data["Phase"]
    Time_Period=data["Time_Period"]
    LowCI=data["LowCI"]
    HighCI=data["HighCI"]
    TPS_Day=data["TPS_Day"]
    TPS_Month=data["TPS_Month"]
    TPS_Year=data["TPS_Year"]
    TPE_Day=data["TPE_Day"]
    TPE_Month=data["TPE_Month"]
    TPE_Year=data["TPE_Year"]
    LowerQR=data["LowerQR"]
    HigherQR=data["HigherQR"]
    #Indic="Took Prescription Medication for Mental Health, Last 4 Weeks"
    Indicator="Indicator_"+data["Indicator"] 
    Group="Group_"+data["Group"]
    State="State_"+data["State"]
    Subgroup="Subgroup_"+data["Subgroup"]

    array=np.zeros(len(ColumnList))

    array[0]=Phase
    array[1]=Time_Period
    array[2]=LowCI
    array[3]=HighCI
    array[4]=TPS_Day
    array[5]=TPS_Month
    array[6]=TPS_Year
    array[7]=TPE_Day
    array[8]=TPE_Month
    array[9]=TPE_Year
    array[10]=LowerQR
    array[11]=HigherQR
    Indicator_index = np.where(ColumnList ==Indicator)[0][0]
    array[Indicator_index] = 1
    Group_index = np.where(ColumnList ==Group)[0][0]
    array[Group_index] = 1
    State_index = np.where(ColumnList ==State)[0][0]
    array[State_index] = 1
    Subgroup_index = np.where(ColumnList ==Subgroup)[0][0]
    array[Subgroup_index] = 1

    #array=array.reshape(1,-1)

    prediction = LinearModel.predict([array])
    prediction = prediction[0]
    

    return jsonify({"Predicted Value":prediction})
        


if __name__ == "__main__":
    app.run(debug = True)



    