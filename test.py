import subprocess
import sys

try:
    import pywt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyWavelets"])
    import pywt  # Now the import should succeed

import speech_recognition as sr
import cv2
import numpy as np
import streamlit as st
import rasterio
from rasterio.plot import reshape_as_image
from skimage.restoration import denoise_nl_means, estimate_sigma
from sklearn.decomposition import PCA
from skimage.feature import peak_local_max
from skimage.filters import rank
from skimage.morphology import disk

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

# Function to read GeoTIFF images
def read_geotiff(filepath):
    with rasterio.open(filepath) as src:
        image = src.read()
        image = reshape_as_image(image)
    return image

# Function to apply image processing with optional parameters and validation
def process_image(image, command, param=None):
    if command == "grayscale":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    elif command == "invert":
        inverted_image = 255 - image
        return inverted_image
    elif command == "blur":
        try:
            strength = int(param[0]) if param else 5
        except (ValueError, TypeError):
            st.warning("Invalid blur strength. Using default (5).")
            strength = 5
        blurred_image = cv2.blur(image, (strength, strength))
        return blurred_image
    elif command == "resize":
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
        try:
            angle = float(param[0]) if param else 0
        except (ValueError, TypeError):
            st.warning("Invalid angle. Using default (0).")
            angle = 0

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        return rotated_image
    elif command in ["add", "subtract", "multiply", "divide"]:
        try:
            value = float(param[0]) if param else 0
        except (ValueError, TypeError):
            st.warning(f"Invalid value for {command}. Using default (0).")
            value = 0

        value_matrix = np.full(image.shape, value, dtype=image.dtype)
        if command == "add":
            result_image = cv2.add(image, value_matrix)
        elif command == "subtract":
            result_image = cv2.subtract(image, value_matrix)
        elif command == "multiply":
            result_image = cv2.multiply(image, value_matrix.astype(image.dtype))
        elif command == "divide":
            if value == 0:
                st.error("Division by zero is not allowed.")
                return image
            result_image = cv2.divide(image.astype(np.float32), value_matrix.astype(np.float32))
            result_image = np.clip(result_image, 0, 255).astype(np.uint8)
        return result_image
    elif command == "histogram":
        if len(image.shape) == 2:
            hist_eq_image = cv2.equalizeHist(image)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            hist_eq_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return hist_eq_image
    elif command == "stitch":
        stitcher = cv2.Stitcher_create()
        status, stitched_image = stitcher.stitch(param)
        if status != cv2.Stitcher_OK:
            st.error("Error stitching images.")
            return None
        return stitched_image
    elif command == "denoise":
        sigma_est = np.mean(estimate_sigma(image))
        denoised_image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)
        denoised_image = (denoised_image * 255).astype(np.uint8)
        return denoised_image
    elif command == "pca":
        pca = PCA(n_components=int(param[0]) if param else 3)
        reshaped_image = image.reshape((-1, 3))
        pca_result = pca.fit_transform(reshaped_image)
        pca_image = pca_result.reshape(image.shape)
        return pca_image
    elif command == "sift":
        sift = cv2.SIFT_create()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        sift_image = cv2.drawKeypoints(gray_image, keypoints, image)
        return sift_image
    elif command == "mean":
        mean_value = np.mean(image)
        st.write(f"Mean value of the image: {mean_value}")
        return image
    elif command == "median":
        median_value = np.median(image)
        st.write(f"Median value of the image: {median_value}")
        return image
    elif command == "max":
        max_value = np.max(image)
        st.write(f"Max value of the image: {max_value}")
        return image
    elif command == "min":
        min_value = np.min(image)
        st.write(f"Min value of the image: {min_value}")
        return image
    elif command == "enhance":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        selem = disk(30)
        enhanced_image = rank.equalize(gray_image, selem=selem)
        return enhanced_image
    elif command == "feature_extraction":
        method = param[0] if param else "sift"
        return feature_extraction(image, method)
    else:
        st.error("Invalid command. Supported commands: grayscale, invert, blur [strength], resize [width height], crop [top left bottom right], rotate [angle], add [value], subtract [value], multiply [value], divide [value], histogram, stitch, denoise, pca [components], sift, mean, median, max, min, enhance, feature_extraction [method]")
        return image

# Function for feature extraction
def feature_extraction(image, method="sift"):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == "sift":
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        feature_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif method == "orb":
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        feature_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif method == "harris":
        gray = np.float32(gray_image)
        harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
        feature_image = image
    else:
        st.error("Unsupported feature extraction method. Supported methods: sift, orb, harris")
        return image

    return feature_image

# Main function for streamlit app
def main():
    st.title("Image Processing App")

    uploaded_file = st.file_uploader("Choose an image file", type=["tif", "tiff", "png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = read_geotiff(uploaded_file) if uploaded_file.name.endswith((".tif", ".tiff")) else np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        if uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        command = get_voice_command()
        if command:
            processed_image = process_image(image.copy(), command[0], command[1:])
            if processed_image is not None:
                st.image(processed_image, caption='Processed Image', use_column_width=True)

if __name__ == "__main__":
    main()
