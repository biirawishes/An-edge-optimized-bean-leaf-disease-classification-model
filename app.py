import gradio as gr
import tensorflow as tf
import numpy as np

# Load model and labels
model = tf.keras.models.load_model('bean_model_full.h5')
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def predict(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array).flatten()
    return {labels[i]: float(prediction[i]) for i in range(len(labels))}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Angular and Rust Bean-Leaf-Disease Model Classifier",
    description="Developed and Built By David Makwetta & Biira Wishes Tricia"
)

interface.launch()