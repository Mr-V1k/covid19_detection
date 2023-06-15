from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    try:
        model = tf.keras.models.load_model('model.h5')
        data = []
        image = request.files['image']
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        data.append(img)
        data = np.array(data)/255.0

        prediction = model.predict(data)

        return render_template('prediction.html', prediction = str(np.argmax(prediction[0])))
    
    except Exception as e:
        app.logger.error(e)
        return render_template("error.html")

if __name__ == '__main__':
    app.run(debug = True)