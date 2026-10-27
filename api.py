from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import uvicorn

app = FastAPI(
    title="API Clasificador de Frutas - MobileNetV2",
    version="2.0",
    description="API REST para clasificar frutas usando Transfer Learning y Fine-Tuning. Convierte imágenes a matriz 128x128x3."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = None
CONFIDENCE_THRESHOLD = 0.60

@app.on_event("startup")
async def load_model():
    global model, class_names
    try:
        model = tf.keras.models.load_model('modelo_frutas_movilenet_finetuned.h5')
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        print(f"Modelo cargado exitosamente. Clases: {class_names}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")

@app.get("/")
def root():
    return {
        "message": "API de Clasificación de Frutas Activa",
        "version": "2.0",
        "endpoints": {
            "/predict": "POST - Clasificar imagen (requiere archivo multipart)",
            "/classes": "GET - Listar clases disponibles",
            "/health": "GET - Verificar estado del servidor"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/classes")
def get_classes():
    if class_names is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    return {"classes": class_names, "total": len(class_names)}

@app.post("/predict")
async def predict_fruit(file: UploadFile = File(...)):
    if model is None or class_names is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # REQUISITO RÚBRICA: Convertir a matriz 128x128x3
        img = img.resize((128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_fruit = class_names[predicted_class_index]
        
        top3_indices = np.argsort(predictions[0])[::-1][:3]
        top3_predictions = [
            {"class": class_names[idx], "confidence": float(predictions[0][idx])}
            for idx in top3_indices
        ]
        
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "status": "no_reconocido",
                "message": "La imagen NO corresponde a las frutas de la base de datos",
                "confidence": confidence,
                "threshold": CONFIDENCE_THRESHOLD,
                "top3_predictions": top3_predictions
            }
        else:
            return {
                "status": "reconocido",
                "predicted_class": predicted_fruit,
                "confidence": confidence,
                "threshold": CONFIDENCE_THRESHOLD,
                "top3_predictions": top3_predictions
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

if __name__ == "__main__":
    print("Iniciando servidor API en http://localhost:8000")
    print("Documentación interactiva disponible en http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)