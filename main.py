import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os
from PIL import Image

# Configure Google Generative AI
genai.configure(api_key="AIzaSyC9ofeMhsLxxB6pw6bENBZUPlveLY_osz0")
os.environ["GOOGLE_API_KEY"] = "AIzaSyC9ofeMhsLxxB6pw6bENBZUPlveLY_osz0"

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model2 = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
    system_instruction=(
        "You are a helpful personal assistant chatbot"
    ),
)

chat = model2.start_chat()

def chat_with_me(question):
    try:
        response = chat.send_message(question)
        return response.text 
    except Exception as e:
        return f"Error: {str(e)}"

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

background_image_url = "https://th.bing.com/th/id/OIP.LAOaWuloBHvVV7ZQRBwcowHaE7?rs=1&pid=ImgDetMain"  # Replace with your background image URL
st.markdown(f"""
    <style>
    .main {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar content above "Connect with Us"
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Chat Support"])

# Other sidebar content...

# Spacer to push "Connect with Us" to the bottom
st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

# "Connect with Us" at the bottom inside the sidebar
st.sidebar.markdown("""
### 
<div style="position: relative; bottom: 0; width: 100%; text-align: center;">
    <h4>Connect with Us</h4>
    <a href="https://github.com/SIT-SIH-2K24/Krishi-Avaranam/tree/master" target="_blank">
        <img src="https://img.icons8.com/material-outlined/24/ffffff/github.png" style="vertical-align: middle;"/>
    </a>
    <a href="https://www.linkedin.com/in/your-linkedin-profile/" target="_blank">
        <img src="https://img.icons8.com/material-outlined/24/ffffff/linkedin.png" style="vertical-align: middle;"/>
    </a>
    <a href="https://www.instagram.com/your-instagram-profile/" target="_blank">
        <img src="https://img.icons8.com/material-outlined/24/ffffff/instagram-new.png" style="vertical-align: middle;"/>
    </a>
</div>
""", unsafe_allow_html=True)




if app_mode == "Home":
    st.markdown("""
    <style>
    .typewriter h1 {
        font-family: 'Courier New', Courier, monospace;
        font-size: 3.5em;
        color: white;  /* Changed to black to contrast with a light background */
        overflow: hidden;
        border-right: .15em solid orange; /* The typewriter cursor */
        white-space: nowrap; /* Keeps the text on a single line */
        margin: 0 auto;
        animation: 
            typing 3.5s steps(40, end),
            blink-caret .75s step-end infinite;
    }
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    @keyframes blink-caret {
        from, to { border-color: black; }
        50% { border-color: black; }
    }
    </style>
    <div class="typewriter">
        <h1>KRISHI AVARANAM</h1>
    </div>
    """, unsafe_allow_html=True)

    # Removed the background image code
    st.markdown("""
    Welcome to KRISHI AVARANAM! 🌿🔍
    
    A AI DRIVEN CROP DISEASE PREDICTION AND MANAGEMENT SYSTEM.

    Our mission is to help in identifying plant diseases efficiently.
    Discover the future of plant disease detection! Upload a plant image, and our state-of-the-art system will rapidly evaluate it for any disease signs. 
    Partner with us to enhance crop health and secure a thriving harvest through innovative, precise analysis. Let’s work together for healthier, more resilient plants.


    ### How It Works
    1. Upload Image: Go to the Disease Recognition page and upload an image of a plant with suspected diseases.
    2. Analysis: Our system will process the image using advanced algorithms to identify potential diseases.
    3. Results: View the results and recommendations for further action.

    ### Why Choose Us?
    - Accuracy: Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - User-Friendly: Simple and intuitive interface for seamless user experience.
    - Fast and Efficient: Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Navigate to the Disease Recognition page in the sidebar to upload your plant image and witness the capabilities of our cutting-edge Plant Disease Recognition System. This powerful tool will analyze your image in-depth, providing you with accurate insights and disease detection. Explore the technology that’s transforming plant health management and optimize your crop care with just a few clicks.

    ### About Us
    Learn more about the project, our team, and our goals on the About page.

    ### Recent Work
    - Successfully integrated Google Generative AI for providing chatbot support within the application.
    - Enhanced the machine learning model for better accuracy and faster predictions.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About the Project
                This project focuses on leveraging machine learning to detect plant diseases from images. It is built using a combination of TensorFlow for model prediction and Google Generative AI for chatbot support. The system is designed to assist farmers and researchers in diagnosing plant health efficiently.
                #### Dataset
                The dataset used in this project is an augmented version of an original dataset, which consists of about 87K RGB images of healthy and diseased crop leaves. These images are categorized into 38 different classes, including various crops and diseases.

                Dataset Structure:
                1. Train: 70295 images
                2. Test: 33 images
                3. Validation: 17572 images

                #### Key Features
                - Advanced ML Models: The project utilizes cutting-edge machine learning models to ensure high accuracy in disease detection.
                - Real-time Chat Support: Integrated Google Generative AI for real-time support, helping users with their queries related to plant diseases.

                #### Achievements
                - Model Optimization: Improved the model's performance and prediction accuracy by fine-tuning the architecture.
                - User Experience: Developed an intuitive interface, making it easy for users to interact with the system.

                #### Future Goals
                - Expand the dataset with more diverse plant species and disease types.
                - Integrate real-time data processing to provide instant feedback on uploaded images.
                - Enhance the chatbot's capabilities to offer more personalized advice and support.
                """)



# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image before attempting to display it.")

    if st.button("Predict"):
        if test_image is not None:
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Reading Labels
            class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                        'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
                        'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 
                        'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
                        'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
                        'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
                        'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
                        'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
                        'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
                        'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
                        'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                        'Tomato___healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
        else:
            st.warning("Please upload an image before attempting to predict.")

# Chat Support Page
elif app_mode == "Chat Support":
    st.header("Agri LifeLine")

    # Initialize session state for chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Function to display chat history
    def display_chat():
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.write(f"You: {msg['content']}")
            else:
                st.write(f"Bot: {msg['content']}")

    # Display existing chat history
    display_chat() 

    # Function to handle sending messages
    def send_message():
        user_message = st.session_state.chat_input
        if user_message:
            st.session_state.messages.append({"role": "user", "content": user_message})
            response = chat_with_me(user_message)
            st.session_state.messages.append({"role": "bot", "content": response})
            # Clear the input field
            st.session_state.chat_input = ""
            # Scroll to bottom
            st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

    # User input with Enter key sending message
    user_input = st.text_input("Type your message here:", key="chat_input", on_change=send_message)
    
    # Send button
    st.button("Send", on_click=send_message)
