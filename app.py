import numpy as np
import pandas as pd
import joblib
import gradio
from xgboost import XGBClassifier



model = joblib.load("xgboost-model.pkl")
# Function for prediction

def predict_death_event(age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time):
    input_df = pd.DataFrame({'age': [float(age)],
                            'anaemia': [int(anaemia)],
                            'creatinine_phosphokinase': [int(creatinine_phosphokinase)],
                            'diabetes': [int(diabetes)],
                            'ejection_fraction': [int(ejection_fraction)],
                            'high_blood_pressure': [int(high_blood_pressure)],
                            'platelets': [float(platelets)],
                            'serum_creatinine': [float(serum_creatinine)],
                            'serum_sodium': [int(serum_sodium)],
                            'sex': [int(sex)],
                            'smoking': [int(smoking)],
                            'time': [int(time)]

    })

    prediction = model.predict(input_df)[0]
    return "Deceased" if prediction == 1 else "Survived"

# Output response
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")

# Gradio interface to generate UI link
def create_interface():
   
    title = "Patient Survival Prediction"
    description = "Predict survival of patient with heart failure, given their clinical record"

    interface = gradio.Interface(fn = predict_death_event,
                         inputs = [gradio.Slider(0.0,100.0,value=1,label="Age"),
                                   gradio.Radio(["Yes","No"],type="index",label="Anaemia"),
                                   gradio.Slider(0,10000,value=50,label="creatinine_phosphokinase"),
                                   gradio.Radio(["Yes","No"],type="index",label="diabetes"),
                                   gradio.Slider(0,100,value=50,label="ejection_fraction"),
                                   gradio.Radio(["Yes","No"],type="index" ,label="high_blood_pressure"),
                                   gradio.Slider(0,950000,value=10000,label="platelets"),
                                   gradio.Slider(0.0,10.0,value=0.5,label="serum_creatinine"),
                                   gradio.Slider(0,150,value=100,label="serum_sodium"),
                                   gradio.Radio(["Male","Female"],type="index",label="Gender"),
                                   gradio.Radio(["Yes","No"],type= "index",label="Smoking"),
                                   gradio.Slider(0,300,value=10,label="Time")],
                         outputs = [out_label],
                         title = title,
                         description = description,
                         allow_flagging='never')
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="127.0.0.1", server_port=7860,share=True, debug=True)
    

