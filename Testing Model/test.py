import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2

# Load the trained model
loaded_model = tf.keras.models.load_model('D:\\Machine Learning Project\\saved_model.h5')

# Define image dimensions
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 60

# Function to preprocess the uploaded image
def preprocess_image(image_data, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)):
    img = cv2.resize(image_data, target_size)
    img = img.astype(np.float32) / 255.0  # Normalize the image
    img = np.transpose(img, (1, 0, 2))  # Transpose dimensions to (60, 80, 3)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to classify the uploaded image
def classify_image(image_data, model):
    prediction = model.predict(image_data)
    return prediction

# Function to get recommendations based on predicted class
def get_recommendations(predicted_class):
    recommendations = []
    # Fetch images from the folder corresponding to the predicted class
    recommended_folder = os.path.join('D:\\Machine Learning Project\\final_Dataset\\test', predicted_class)
    print("Recommended folder:", recommended_folder)  # Debugging statement
    if os.path.exists(recommended_folder):
        recommended_images = os.listdir(recommended_folder)
        np.random.shuffle(recommended_images)  # Shuffle to get random recommendations
        for img_file in recommended_images[:5]:  # Select top 5 recommendations
            img_path = os.path.join(recommended_folder, img_file)
            recommendations.append(img_path)
    else:
        print("Folder not found:", recommended_folder)  # Debugging statement
    return recommendations

# Streamlit app
def main():
    st.title('Image Recommendation App')

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.subheader('Uploaded Image:')
        st.image(uploaded_file, caption='Uploaded Image', width=200)  # Adjust width as needed

        # Convert uploaded image to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Preprocess uploaded image
        preprocessed_image = preprocess_image(opencv_image)

        # Classify uploaded image
        prediction = classify_image(preprocessed_image, loaded_model)
        class_names = ['Accessories', 'Apparel', 'Personal Care']  # Define your class names
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]

        st.subheader('Predicted Class:')
        st.write(predicted_class)

        # Get recommendations based on predicted class
        recommendations = get_recommendations(predicted_class)
        print("Recommendations:", recommendations)  # Debugging statement

        # Display recommended images with probabilities
        st.subheader('Recommended Images:')
        if recommendations:
            for img_path in recommendations:
                st.write(f'Recommended Image ({predicted_class}):')
                img = cv2.imread(img_path)
                # Resize recommended image to 80x60 pixels
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                st.image(img, caption=img_path, channels="BGR")
                st.write("")  # Add some space between images
        else:
            st.write("No recommendations found.")

        # Compute probability of the most similar image
        if recommendations:
            most_similar_img_path = recommendations[0]
            most_similar_prediction = classify_image(preprocess_image(cv2.imread(most_similar_img_path)), loaded_model)
            st.write("Probability of most similar image:", most_similar_prediction[0][predicted_class_index])

if __name__ == '__main__':
    main()
