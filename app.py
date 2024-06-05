from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import speech_recognition as sr
import cv2
import numpy as np
from gtts import gTTS
from playsound3 import playsound

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Please say something...")
    speak_text("Please say something...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        speak_text("Recognizing...")
        text = recognizer.recognize_google(audio)
        speak_text(f"Recognized text: {text}")
        return text
    except sr.RequestError:
        speak_text("API unavailable")
        print("API unavailable")
        return None
    except sr.UnknownValueError:
        speak_text("Unable to recognize speech")
        print("Unable to recognize speech")
        return None

def extract_parameters_from_text(text):
    words = text.split()
    if len(words) >= 2:
        try:
            scale_factor = float(words[0])
            angle = float(words[1])
            return scale_factor, angle
        except ValueError:
            print("Could not parse scale factor and angle from text")
            return 1.0, 0.0
    else:
        print("Not enough information in text")
        return 1.0, 0.0

def transform_image(image_path, scale_factor, angle):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    center = (new_width // 2, new_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(scaled_image, rotation_matrix, (new_width, new_height))
    return rotated_image

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    playsound("response.mp3")
    os.remove("response.mp3")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('transform', filename=file.filename))
    return render_template('index.html')

@app.route('/transform/<filename>', methods=['GET', 'POST'])
def transform(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if request.method == 'POST':
        recognized_text = recognize_speech_from_microphone()
        if recognized_text:
            scale_factor, angle = extract_parameters_from_text(recognized_text)
            transformed_image = transform_image(filepath, scale_factor, angle)
            output_filename = f"transformed_{filename}"
            output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_filepath, transformed_image)
            return send_from_directory(app.config['OUTPUT_FOLDER'], output_filename)
    return render_template('transform.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
