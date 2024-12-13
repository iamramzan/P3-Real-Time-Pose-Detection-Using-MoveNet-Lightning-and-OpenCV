# P8: Real-Time Pose Detection Using MoveNet Lightning and OpenCV
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Load the TensorFlow Lite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

# Function to draw keypoints (joints) on the frame
def draw_keypoints(frame, keypoints, confidence_threshold):
    """
    Draws keypoints on the frame if their confidence exceeds the threshold.
    - frame: The image on which keypoints are drawn.
    - keypoints: The predicted keypoints from the model.
    - confidence_threshold: The minimum confidence value for rendering keypoints.
    """
    y, x, c = frame.shape  # Get the frame dimensions
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # Scale keypoints to image size
    
    for kp in shaped:
        ky, kx, kp_conf = kp  # Extract coordinates and confidence
        if kp_conf > confidence_threshold:  # Only draw keypoints above threshold
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)  # Draw a green circle

# Define connections between keypoints and their colors
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', (0, 5): 'm',
    (0, 6): 'c', (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c',
    (5, 6): 'y', (5, 11): 'm', (6, 12): 'c', (11, 12): 'y',
    (11, 13): 'm', (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'
}

# Function to draw connections (bones) between keypoints
def draw_connections(frame, keypoints, edges, confidence_threshold):
    """
    Draws connections (edges) between keypoints if both keypoints have confidence 
    above the threshold.
    - frame: The image on which connections are drawn.
    - keypoints: The predicted keypoints from the model.
    - edges: A dictionary mapping pairs of keypoints to their colors.
    - confidence_threshold: The minimum confidence value for rendering connections.
    """
    y, x, c = frame.shape  # Get the frame dimensions
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # Scale keypoints to image size
    
    for edge, color in edges.items():  # Loop through each connection
        p1, p2 = edge  # Get the indices of the two keypoints
        y1, x1, c1 = shaped[p1]  # First keypoint coordinates and confidence
        y2, x2, c2 = shaped[p2]  # Second keypoint coordinates and confidence
        
        # Draw a line only if both keypoints exceed the confidence threshold
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red line

# Initialize webcam capture
cap = cv2.VideoCapture(1)  # Open webcam (use '0' if there's only one camera)
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Exit loop if no frame is captured
    
    # Preprocess the frame for the model
    img = frame.copy()  # Make a copy of the frame
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)  # Resize with padding
    input_image = tf.cast(img, dtype=tf.float32)  # Convert to float32 for the model
    
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Run inference on the input image
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()  # Invoke the model
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])  # Get predictions
    
    # Render keypoints and connections on the frame
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)  # Draw connections
    draw_keypoints(frame, keypoints_with_scores, 0.4)  # Draw keypoints
    
    # Display the output frame
    cv2.imshow('MoveNet Lightning', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
