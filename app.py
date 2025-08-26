from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from model.predict import analyze_images
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key_here'  # needed for flash messages

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['brain_tumor_db']
patients_collection = db['patients']

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    patient_name = request.form.get('patient_name')
    age = request.form.get('age')
    gender = request.form.get('gender')

    uploaded_files = []
    for position in ['top', 'bottom', 'left', 'right']:
        file = request.files.get(position)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)

    # Analyze the MRI images
    result, confidence = analyze_images(uploaded_files)

    # Save to MongoDB
    patient_record = {
        'name': patient_name,
        'age': age,
        'gender': gender,
        'result': result,
        'confidence': confidence,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
    }
    patients_collection.insert_one(patient_record)

    return render_template('report.html',
                           patient_name=patient_name,
                           age=age,
                           gender=gender,
                           result=result,
                           confidence=confidence)

@app.route('/patients')
def list_patients():
    patients = list(patients_collection.find())
    return render_template('patients.html', patients=patients)

@app.route('/patient/<patient_id>')
def patient_detail(patient_id):
    try:
        patient = patients_collection.find_one({'_id': ObjectId(patient_id)})
    except Exception:
        patient = None
    if patient:
        return render_template('report.html',
                               patient_name=patient['name'],
                               age=patient['age'],
                               gender=patient['gender'],
                               result=patient['result'],
                               confidence=patient['confidence'])
    return "Patient not found", 404

@app.route('/delete_patient/<patient_id>', methods=['POST'])
def delete_patient(patient_id):
    try:
        patients_collection.delete_one({'_id': ObjectId(patient_id)})
        flash("Patient record deleted successfully.", "success")
    except Exception as e:
        flash("Failed to delete patient record.", "error")
    return redirect(url_for('list_patients'))

if __name__ == '__main__':
    app.run(debug=True)
