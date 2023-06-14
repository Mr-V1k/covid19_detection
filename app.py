from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    image = request.files['image']

    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return render_template('prediction.html', prediction = str(np.argmax(prediction[0])))

if __name__ == '__main__':
    app.run(debug = True)