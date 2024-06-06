import speech_recognition as sr
import cv2
import numpy as np
import streamlit as st

# Function to capture and recognize speech (with follow-up prompt and retry)
def get_voice_command(retries=3):
    r = sr.Recognizer()
    for attempt in range(retries):
        with sr.Microphone() as source:
            st.write("Listening for command...")
            audio = r.listen(source)
        try:
            command = r.recognize_google(audio)
            st.write("Command:", command)
            return command.lower().split()  # Split on spaces for potential parameters
        except sr.UnknownValueError:
            st.write("Could not understand audio. Trying again...")
            if attempt == retries - 1:
                st.error("Failed to recognize voice command after multiple attempts.")
                return None
        except sr.RequestError as e:
            st.error("Could not request results from Google Speech Recognition service; {0}".format(e))
            return None

# Function to apply image processing with optional parameters and validation
def process_image(image, command, param=None):
    if command == "grayscale":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    elif command == "invert":
        inverted_image = 255 - image
        return inverted_image
    elif command == "blur":
        # Handle optional blur strength parameter (default 5)
        try:
            strength = int(param[0]) if param else 5
        except (ValueError, TypeError):
            st.warning("Invalid blur strength. Using default (5).")
            strength = 5
        blurred_image = cv2.blur(image, (strength, strength))
        return blurred_image
    elif command == "resize":
        # Handle optional width and height parameters, validate within image dimensions
        try:
            width, height = (int(param[0]), int(param[1])) if param and len(param) == 2 else (image.shape[1], image.shape[0])
            if width <= 0 or height <= 0:
                raise ValueError
        except (ValueError, TypeError):
            st.warning("Invalid width or height. Using original size.")
            width, height = image.shape[1], image.shape[0]
        resized_image = cv2.resize(image, (width, height))
        return resized_image
    elif command == "crop":
        # Handle optional top, left, bottom, right coordinates, validate within image dimensions
        try:
            top, left, bottom, right = (int(param[0]), int(param[1]), int(param[2]), int(param[3])) if param and len(param) == 4 else (0, 0, image.shape[0], image.shape[1])
            if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
                raise ValueError
        except (ValueError, TypeError):
            st.warning("Invalid crop coordinates. Using entire image.")
            top, left, bottom, right = 0, 0, image.shape[0], image.shape[1]
        cropped_image = image[top:bottom, left:right]
        return cropped_image
    elif command == "rotate":
        # Handle optional angle parameter, validate it's a number
        try:
            angle = float(param[0]) if param else 0
        except (ValueError, TypeError):
            st.warning("Invalid angle. Using default (0).")
            angle = 0

        # Compute the center of the image
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute the new bounding dimensions of the image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Adjust the rotation matrix to take into account the new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform the actual rotation and return the image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        return rotated_image
    else:
        st.error("Invalid command. Supported commands: grayscale, invert, blur [strength], resize [width height], crop [top left bottom right], rotate [angle]")
        return image

# Main app
def main():
    """Image processing app with voice commands or text input"""

    st.title("Image Processing with Voice Commands or Text Input")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")

        # Command options
        command_options = ["grayscale", "invert", "blur", "resize", "crop", "rotate"]
        command_text = st.text_input("Enter command details (optional):")
        use_voice_command = st.button("Use voice command")

        command, param = None, None

        # Extract command and parameter from voice recognition or text input
        if use_voice_command:
            command_parts = get_voice_command()  # Retry on failures
            if command_parts:
                command = command_parts[0]
                param = command_parts[1:] if len(command_parts) > 1 else None
        elif command_text:
            command_parts = command_text.lower().split()
            command = command_parts[0]
            param = command_parts[1:] if len(command_parts) > 1 else None

        # Process image if a valid command is received
        if command:
            processed_image = process_image(image.copy(), command, param)
            if processed_image is not None:
                st.write("Processed Image:")
                if len(processed_image.shape) == 2:  # Check if the image is grayscale
                    st.image(processed_image, channels="GRAY")
                else:
                    st.image(processed_image, channels="BGR")

                # Download button for processed image
                if st.button("Download Processed Image"):
                    cv2.imwrite("processed_image.jpg", processed_image)
                    with open("processed_image.jpg", "rb") as file:
                        btn = st.download_button(
                            label="Download Processed Image",
                            data=file,
                            file_name="processed_image.jpg",
                            mime="image/jpeg"
                        )
                        st.success("Processed image ready for download.")
            else:
                st.error("Error processing image. Please try again.")

if __name__ == "__main__":
    main()
