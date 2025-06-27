import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import load_model
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import json
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, send_file
import base64
from io import BytesIO
from data.student_data import STUDENT_DATA


class_names = ['Abhinav', 'Adinath', 'Aditya', 'Aishwarya', 'Alana', 'Albin', 'Alen Joseph', 'Alen K Abraham', 'Alna Jas', 'Amal Naveen', 'Aman Siddeeque', 'Aman T Shekar', 'Angith', 'Anjana', 'Anuvinda', 'Ardra', 'Arshad', 'Aswin', 'Athul Krishna', 'FIjul', 'Fathah', 'Fathima Z', 'Fathimath Shahsoora', 'Gowthami', 'Jasira', 'Jyothi', 'Mohammed Mazin', 'Mohammed Rahil Sulaiman', 'Mohammed Salih', 'Muhammed Jishan', 'Parvathy', 'Sabeeh', 'Sivani', 'Sourav', 'Sreelakshmi GC', 'Sreelakshmi PS', 'Sreenandana Babu', 'Vishnu', 'Vismaya']


tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


model_path = "models/face_recognition_model.h5"
label_encoder_path = "models/label_encoder.pkl"
signature_model_path = "models/identity_model.tflite"
signature_labels_path = "models/signature_labels.txt"


try:
    
    signature_interpreter = tf.lite.Interpreter(model_path=signature_model_path)
    signature_interpreter.allocate_tensors()
    signature_input_details = signature_interpreter.get_input_details()
    signature_output_details = signature_interpreter.get_output_details()
    signature_img_height, signature_img_width = signature_input_details[0]['shape'][1:3]
    
    
    try:
        with open(signature_labels_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(class_names)} signature classes")
    except Exception as label_error:
        print(f"Warning: Could not load signature labels from file: {label_error}")
        print("Using default class names")
        
except Exception as e:
    print(f"Error loading signature model: {e}")


def create_model():
    model = Sequential([
        Input(shape=(128,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(39, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


try:
    model = create_model()
    model.load_weights(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model architecture matches the saved weights")


try:
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading label encoder: {e}")


ANSWER_KEY_FILE = "data/answer_key.json"

def load_answer_key():
    try:
        if os.path.exists(ANSWER_KEY_FILE):
            with open(ANSWER_KEY_FILE, 'r') as f:
                return json.load(f)
        return {
            "1": 2,  
            "2": 2,  
            "3": 3,  
            "4": 4,  
            "5": 3,  
            "6": 3,  
            "7": 2,  
            "8": 2,  
            "9": 2,  
            "10": 2  
        }
    except Exception as e:
        print(f"Error loading answer key: {e}")
        return {}

def save_answer_key(answer_key):
    try:
        os.makedirs(os.path.dirname(ANSWER_KEY_FILE), exist_ok=True)
        with open(ANSWER_KEY_FILE, 'w') as f:
            json.dump(answer_key, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving answer key: {e}")
        return False


answer_key = load_answer_key()


def extract_face(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    face_crop = image[0:int(height * 0.15), int(width*0.05):int(width * 0.22)]
    face_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_face.jpg")
    cv2.imwrite(face_path, face_crop)
    return face_path

def predict_identity(face_path):
    embedding_list = DeepFace.represent(img_path=face_path, model_name="Facenet", enforce_detection=False)
    if embedding_list:
        embedding = embedding_list[0]['embedding']
        embedding = np.array(embedding).reshape(1, -1)
        prediction = model.predict(embedding)
        predicted_label_index = np.argmax(prediction)
        confidence = prediction[0][predicted_label_index]
        if confidence > 0.5:
            return label_encoder.inverse_transform([predicted_label_index])[0]
    return "Unknown"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]
    
    
    cv2.imwrite('debug/debug_gray.jpg', gray)
    cv2.imwrite('debug/debug_thresh.jpg', thresh)
    
    return image, thresh

def detect_bubbles(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\nInitial contours found: {len(contours)}")
    
    
    bubble_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 100 < area < 1000:
            x, y, w, h = cv2.boundingRect(c)
            bubble_contours.append((c, (x, y)))
            print(f"Found bubble: Area={area:.1f}, Position=({x}, {y})")
    
    
    bubble_contours.sort(key=lambda x: (x[1][1], x[1][0]))  
    
    
    questions = []
    current_row = []
    last_y = -1
    
    for contour, (x, y) in bubble_contours:
        if last_y == -1:
            last_y = y
        
        
        if abs(y - last_y) > 20:  
            if current_row:
                
                current_row.sort(key=lambda c: cv2.boundingRect(c[0])[0])
                
                current_row = [c[0] for c in current_row]
                questions.append(current_row)
            current_row = [(contour, (x, y))]
            last_y = y
        else:
            current_row.append((contour, (x, y)))
    
    
    if current_row:
        current_row.sort(key=lambda c: cv2.boundingRect(c[0])[0])
        current_row = [c[0] for c in current_row]
        questions.append(current_row)
    
    
    questions = questions[:10]
    
    print(f"\nBubble Detection Summary:")
    print(f"Total contours detected: {len(contours)}")
    print(f"Filtered bubble contours: {len(bubble_contours)}")
    print(f"Questions formed: {len(questions)}")
    
    
    for i, q in enumerate(questions):
        print(f"Question {i+1}: {len(q)} options")
        
        x_coords = [cv2.boundingRect(c)[0] for c in q]
        print(f"  X-coordinates: {x_coords}")
    
    return questions

def identify_filled_bubbles(thresh, questions):
    filled_bubbles = {}
    print("\nBubble Fill Analysis:")
    
    for q_index, options in enumerate(questions, start=1):
        filled = -1
        max_intensity = 0
        intensities = []
        positions = []
        
        print(f"\nQuestion {q_index}:")
        
        options_with_pos = [(opt, cv2.boundingRect(opt)[0]) for opt in options]
        options_with_pos.sort(key=lambda x: x[1])
        options = [opt for opt, _ in options_with_pos]
        
        for opt_index, contour in enumerate(options):
            x, y, w, h = cv2.boundingRect(contour)
            roi = thresh[y:y+h, x:x+w]
            filled_ratio = cv2.countNonZero(roi) / (w * h)
            intensities.append(filled_ratio)
            positions.append((x, y))
            
            print(f"  Option {chr(65+opt_index)}: Position=({x}, {y}), "
                  f"Size={w}x{h}, Fill Ratio={filled_ratio:.3f}")
            
            if filled_ratio > 0.2 and filled_ratio > max_intensity:
                max_intensity = filled_ratio
                filled = opt_index + 1
        
        filled_bubbles[str(q_index)] = filled
        print(f"  Selected: {chr(64+filled) if filled > 0 else 'None'} "
              f"(Fill Ratio: {max_intensity:.3f})")
        print(f"  All Intensities: {[f'{i:.3f}' for i in intensities]}")
    
    return filled_bubbles

def grade_omr(filled_bubbles, answer_key):
    score = 0
    print("\nGrading Details:")
    for q_num, correct_ans in answer_key.items():
        student_ans = filled_bubbles.get(str(q_num), -1)
        
        student_ans = int(student_ans) if student_ans != -1 else -1
        correct_ans = int(correct_ans)
        is_correct = student_ans == correct_ans
        print(f"Q{q_num}: Student={chr(64+student_ans) if student_ans > 0 else 'None'}, "
              f"Correct={chr(64+correct_ans)}, Match={is_correct}")
        if is_correct:
            score += 1
    print(f"Total Score: {score}/{len(answer_key)}")
    return score

def process_omr(image_path, answer_key):
    print("\n=== Starting OMR Processing ===")
    print(f"Processing image: {image_path}")
    
    image, thresh = preprocess_image(image_path)
    questions = detect_bubbles(thresh)
    filled_bubbles = identify_filled_bubbles(thresh, questions)
    score = grade_omr(filled_bubbles, answer_key)
    
    
    result_image = image.copy()
    debug_image = image.copy()  
    
    for q_index, options in enumerate(questions, start=1):
        q_str = str(q_index)
        for opt_index, contour in enumerate(options):
            x, y, w, h = cv2.boundingRect(contour)
            
            
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(debug_image, f"Q{q_index}{chr(65+opt_index)}", (x-20, y+h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            
            cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 2)
            
            if (opt_index + 1) == filled_bubbles[q_str]:
                color = (0, 255, 0) if filled_bubbles[q_str] == answer_key[q_str] else (0, 0, 255)
                cv2.fillPoly(result_image, [contour], color)
            
            cv2.putText(result_image, chr(65 + opt_index), (x-20, y+h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    
    cv2.imwrite('debug/debug_detection.jpg', debug_image)
    cv2.imwrite('debug/debug_result.jpg', result_image)
    
    print("\n=== OMR Processing Complete ===")
    return filled_bubbles, score, result_image

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def extract_signature(image_path):
    """Extracts the signature from an image using OpenCV"""
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    
    height, width = image.shape[:2]
    
    
    signature_region = image[:int(height * 0.2), int(width * 0.6):]  
    
    
    gray = cv2.cvtColor(signature_region, cv2.COLOR_BGR2GRAY)
    
    
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No signature detected in the image")
    
    
    max_contour = max(contours, key=cv2.contourArea)
    
    
    x, y, w, h = cv2.boundingRect(max_contour)
    
    
    signature = signature_region[y:y+h, x:x+w]
    
    
    debug_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_signature.png')
    debug_region_path = os.path.join(app.config['UPLOAD_FOLDER'], 'signature_region.png')
    debug_thresh_path = os.path.join(app.config['UPLOAD_FOLDER'], 'signature_thresh.png')
    
    cv2.imwrite(debug_path, signature)
    cv2.imwrite(debug_region_path, signature_region)
    cv2.imwrite(debug_thresh_path, thresh)
    
    return signature

def preprocess_signature(signature):
    """Preprocess extracted signature for the model"""
    
    img = cv2.cvtColor(signature, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (224, 224))
    
    img = (img.astype(np.float32) / 127.5) - 1.0
    
    img = np.expand_dims(img, axis=0)
    return img

def predict_signature(signature):
    """Run inference on extracted signature"""
    img = preprocess_signature(signature)
    signature_interpreter.set_tensor(signature_input_details[0]['index'], img)
    signature_interpreter.invoke()
    output = signature_interpreter.get_tensor(signature_output_details[0]['index'])[0]
    
    if len(output) > 1:  
        predicted_index = np.argmax(output)
        confidence = output[predicted_index]
        
        print(f"\nSignature Prediction Debug:")
        print(f"Output shape: {output.shape}")
        print(f"Number of classes in model output: {len(output)}")
        print(f"Number of available class names: {len(class_names)}")
        print(f"Predicted Index: {predicted_index}")
        print(f"Raw confidence scores:", output)
        print(f"Max confidence: {confidence}")
        
        
        top_3_indices = np.argsort(output)[-3:][::-1]
        print("\nTop 3 predictions:")
        for idx in top_3_indices:
            if idx < len(class_names):
                print(f"Class: {class_names[idx]}, Confidence: {output[idx]:.4f}")
            else:
                print(f"Warning: Index {idx} out of range for class_names")
    else:  
        predicted_index = int(output[0] >= 0.5)
        confidence = output[0]
        print(f"Binary Classification - Confidence: {confidence}")
    
    
    if predicted_index >= len(class_names):
        print(f"Warning: Predicted index {predicted_index} is out of range. Defaulting to Unknown")
        return 0, 0.0  
    
    return predicted_index, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer_key')
def answer_key_page():
    return render_template('answer_key.html', answer_key=answer_key)

@app.route('/update_answer_key', methods=['POST'])
def update_answer_key():
    try:
        new_answer_key = json.loads(request.form.get('answer_key', '{}'))
        
        
        if not isinstance(new_answer_key, dict):
            return jsonify({'success': False, 'error': 'Invalid answer key format'})
            
        for q_num, answer in new_answer_key.items():
            if not q_num.isdigit() or int(q_num) < 1 or int(q_num) > 10:
                return jsonify({'success': False, 'error': f'Invalid question number: {q_num}'})
            if not isinstance(answer, int) or answer < 1 or answer > 4:
                return jsonify({'success': False, 'error': f'Invalid answer for question {q_num}'})
        
        
        global answer_key
        answer_key = new_answer_key
        if save_answer_key(answer_key):
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to save answer key to file'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/signature')
def signature_page():
    return render_template('signature.html')

@app.route('/verify_signature', methods=['POST'])
def verify_signature():
    try:
        if 'signature_image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['signature_image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_signature.jpg')
            file.save(filename)
            
            try:
                print("\nProcessing signature verification...")
                
                
                signature = cv2.imread(filename)
                if signature is None:
                    raise ValueError("Error loading signature image")
                
                
                signature = cv2.resize(signature, (224, 224))
                signature = cv2.cvtColor(signature, cv2.COLOR_BGR2RGB)
                signature = (signature.astype(np.float32) / 127.5) - 1.0
                signature = np.expand_dims(signature, axis=0)
                
                
                signature_interpreter.set_tensor(signature_input_details[0]['index'], signature)
                signature_interpreter.invoke()
                output = signature_interpreter.get_tensor(signature_output_details[0]['index'])[0]
                
                predicted_index = np.argmax(output)
                confidence = output[predicted_index]
                
                signature_name = class_names[predicted_index] if confidence > 0.15 else "Unknown"
                
                
                preview_image = cv2.imread(filename)
                preview_image = cv2.resize(preview_image, (400, 200))  
                preview_base64 = encode_image(preview_image)
                
                
                os.remove(filename)
                
                print(f"Signature verification completed. Name: {signature_name}, Confidence: {confidence}")
                
                return jsonify({
                    'signature_name': signature_name,
                    'confidence': float(confidence),
                    'preview_image': f'data:image/jpeg;base64,{preview_base64}'
                })
                
            except Exception as e:
                print(f"Error during signature verification: {str(e)}")
                import traceback
                print("Full traceback:")
                print(traceback.format_exc())
                return jsonify({'error': f'Error verifying signature: {str(e)}'}), 500
            finally:
                
                if 'filename' in locals() and os.path.exists(filename):
                    os.remove(filename)
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'omr_sheet' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        register_number = request.form.get('register_number', '').strip().upper()
        if not register_number:
            return jsonify({'error': 'Register number is required'}), 400
        
        if register_number not in STUDENT_DATA:
            return jsonify({'error': 'Invalid register number'}), 400
        
        file = request.files['omr_sheet']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_omr.jpg')
            file.save(filename)
            
            try:
                print(f"\nProcessing OMR sheet for register number: {register_number}")
                
                
                print("Extracting face from image...")
                face_path = extract_face(filename)
                candidate_name = predict_identity(face_path)
                print(f"Detected face name: {candidate_name}")
                
                
                print("Processing OMR sheet...")
                filled_bubbles, score, result_image = process_omr(filename, answer_key)
                print(f"OMR score: {score}/{len(answer_key)}")
                
                
                print("Encoding result image...")
                result_image_base64 = encode_image(result_image)
                
                
                print("Cleaning up temporary files...")
                os.remove(face_path)
                os.remove(filename)
                
                print("Processing completed successfully")
                
                return jsonify({
                    'register_number': register_number,
                    'registered_name': STUDENT_DATA[register_number],
                    'candidate_name': candidate_name,
                    'score': score,
                    'total_questions': len(answer_key),
                    'answers': filled_bubbles,
                    'result_image': f'data:image/jpeg;base64,{result_image_base64}'
                })
                
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                import traceback
                print("Full traceback:")
                print(traceback.format_exc())
                return jsonify({'error': f'Error processing OMR sheet: {str(e)}'}), 500
            finally:
                
                try:
                    if 'face_path' in locals() and os.path.exists(face_path):
                        os.remove(face_path)
                    if 'filename' in locals() and os.path.exists(filename):
                        os.remove(filename)
                except Exception as cleanup_error:
                    print(f"Error during cleanup: {str(cleanup_error)}")
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
