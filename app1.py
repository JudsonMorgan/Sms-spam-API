from fastapi import FastAPI
import joblib

model = joblib.load("Models/model.sav")
app = FastAPI(title="An API to predict SPAM and HAM messages", description="type a sentence and predict whether it is a spam or ham texts")
@app.post("/predict", summary="Predicts whether a text message is spam or ham", tags="Spam or Ham")
def predict(data : str)-> str:
    return model.predict([data])[0]
    