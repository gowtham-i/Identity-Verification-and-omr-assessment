import tensorflow as tf
import numpy as np
import cv2
import sys

# Load class names (modify if stored in a file)
class_names = ['Abhinav', 'Adinath', 'Aditya', 'Aishwarya', 'Alana', 'Albin', 'Alen Joseph', 'Alen K Abraham', 'Alna Jas', 'Amal Naveen', 'Aman Siddeeque', 'Aman T Shekar', 'Angith', 'Anjana', 'Anuvinda', 'Ardra', 'Arshad', 'Aswin', 'Athul Krishna', 'FIjul', 'Fathah', 'Fathima Z', 'Fathimath Shahsoora', 'Gowthami', 'Jasira', 'Jyothi', 'Mohammed Mazin', 'Mohammed Rahil Sulaiman', 'Mohammed Salih', 'Muhammed Jishan', 'Parvathy', 'Sabeeh', 'Sivani', 'Sourav', 'Sreelakshmi GC', 'Sreelakshmi PS', 'Sreenandana Babu', 'Vishnu', 'Vismaya']

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/identity_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_height, img_width = input_details[0]['shape'][1:3]

def extract_signature(image_path):
    """ Extracts the signature from an image using OpenCV """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour, assuming it's the signature
    if contours:
        signature_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(signature_contour)
        signature = img[y:y+h, x:x+w]
        return signature
    else:
        print("No signature detected.")
        sys.exit(1)

def preprocess_image(image):
    """ Preprocess extracted signature for the model """
    img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    img = cv2.resize(img, (img_width, img_height))
    img = (img.astype(np.float32) / 127.5) - 1.0  # Normalize to [-1, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict(image_path):
    """ Run inference on extracted signature """
    signature = extract_signature(image_path)
    img = preprocess_image(signature)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    if len(output) > 1:  # Multi-class classification
        predicted_index = np.argmax(output)
        confidence = output[predicted_index]
    else:  # Binary classification
        predicted_index = int(output[0] >= 0.5)
        confidence = output[0]
    
    predicted_class = class_names[predicted_index]
    print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict(image_path)
