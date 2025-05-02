import numpy as np
import pandas as pd
import joblib
import gradio
from xgboost import XGBClassifier
import prometheus_client as prom
from sklearn.metrics import r2_score
from fastapi import FastAPI, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST,start_http_server, Gauge

from starlette.responses import Response
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
curr_path = str(Path(__file__).parent)

app=FastAPI()


# Metric object of type gauge
avg_prediction_gauge = prom.Gauge('model_prediction_avg', 'Average prediction over 100 samples')

#LOAD model
model = joblib.load("xgboost-model.pkl")


# --------------------------
# Prometheus metrics endpoint
# --------------------------

@app.get("/metrics")
async def get_metrics():
        update_metrics()
        return Response(media_type="text/plain", content= prom.generate_latest())
   


def update_metrics():

    # LOAD TEST DATA
    df = pd.read_csv(curr_path +"/heart_failure_clinical_records_dataset.csv")
    df=df.head(100)

    X = df.drop(columns=['DEATH_EVENT']).values
    y_true = df['DEATH_EVENT'].values

    y_pred = model.predict(X)

    # Calculate metrics
    avg_pred = np.mean(y_pred)
          

    # Update Prometheus metrics
    avg_prediction_gauge.set(avg_pred)
    
           
   

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
   
    return prediction
    #return "Deceased" if prediction == 1 else "Survived"



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


# Mount gradio interface object on FastAPI app at endpoint = '/'
interface = create_interface()
app = gradio.mount_gradio_app(app, interface, path="/")


if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
    
  
    

    

