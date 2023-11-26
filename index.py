import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, jsonify, request, render_template
import io, base64
from PIL import Image

app = Flask(__name__, static_folder='static')

cascade_faces = 'assets/haarcascade_frontalface_default.xml'
caminho_modelo = 'assets/modelo_01_expressoes.h5'
expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]

face_detection = cv2.CascadeClassifier(cascade_faces)
classificador_emocoes = load_model(caminho_modelo, compile=False)

def detectar_emocoes(imagem):
    original = imagem.copy()
    cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # @param cinza: A imagem em escala de cinza na qual você deseja realizar a detecção de faces.
    # @param scaleFactor: Este parâmetro compensa a redução da taxa de escala entre os tamanhos da 
    #                     imagem. Ele é usado para criar uma pirâmide de imagens em diferentes escalas 
    #                     para detecção de objetos em várias escalas. Um valor mais baixo tornará a detecção mais sensível, mas também mais lenta.
    # @param minNeighbors: Número de vizinhos que cada candidato a retângulo de face deve ter para ser considerado parte de uma face. 
    #                      Este parâmetro ajuda a filtrar falsos positivos. Valores mais altos resultam em uma detecção mais conservadora.
    # @param minSize: Tamanho mínimo da face possível. Qualquer retângulo menor que isso é ignorado. 
    #                 Isso ajuda a remover detecções insignificantes e a melhorar a precisão.
    faces = face_detection.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

    if len(faces) > 0:
        fX, fY, fW, fH = faces[0]
        roi = cinza[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = classificador_emocoes.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = expressoes[preds.argmax()]

        return label, float(emotion_probability), faces, preds

    else:
        return '', 0, np.array([]), np.array([])

def convert_image_to_numpy_array(image_data):
  return np.asarray(image_data)


@app.post("/face")
def getEmotion():
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(request.json.get('base64'), "utf-8"))))

    img = convert_image_to_numpy_array(img)

    # Convertendo formato de cor da imagem
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    label, emotion_probability, faces, preds = detectar_emocoes(img)
    return jsonify({ 'label': label, 'emotion_probability': emotion_probability, 'faces': faces.tolist(), 'preds': preds.tolist() })

@app.get("/")
def getIndex():
    return render_template("index.html")

@app.get("/recognition/realtime")
def getRecognitionRealtime():
    return render_template("recognition-realtime.html")

@app.get("/recognition/image")
def getRecognitionImage():
    return render_template("recognition-image.html")