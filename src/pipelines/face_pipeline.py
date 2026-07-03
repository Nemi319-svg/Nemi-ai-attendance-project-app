import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students

@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector() 

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec

def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, 1)

    encodings = []

    for face in faces:
        shape = sp(image_np, face)
        # Generate 128-dimensional embedding
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1) 
        encodings.append(np.array(face_descriptor))
        
    return encodings

@st.cache_resource
def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None
    
    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))

    if len(X) == 0:
        return None
    
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')

    try:
        clf.fit(X, y)
    except ValueError:
        pass

    return {'clf': clf, 'X': X, "y": y}

def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)

def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)
    detected_student = {}

    model_data = get_trained_model()

    if not model_data:
        return detected_student, [], len(encodings)
    
    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))
    
    # Stricter distance threshold to prevent cross-user false positives
    RESEMBLANCE_THRESHOLD = 0.45 

    for encoding in encodings:
        # 1. Determine the suspected ID using SVM (if multiple users exist)
        if len(all_students) >= 2:
            predicted_id = int(clf.predict([encoding])[0])
        else:
            predicted_id = int(all_students[0])

        # 2. Extract ALL embeddings belonging to this specific student
        student_embeddings = [X_train[i] for i, uid in enumerate(y_train) if uid == predicted_id]

        # 3. Calculate Euclidean distance to all their registered photos
        distances = [np.linalg.norm(emb - encoding) for emb in student_embeddings]
        
        # 4. Find the closest match score among their photos
        best_match_score = min(distances) if distances else float('inf')

        # 5. Strict Validation
        if best_match_score <= RESEMBLANCE_THRESHOLD:
            detected_student[predicted_id] = True

    return detected_student, all_students, len(encodings)
