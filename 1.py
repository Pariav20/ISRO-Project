import cv2
import numpy as np
import streamlit as st
from skimage import exposure
from skimage.restoration import denoise_nl_means, estimate_sigma
from sklearn.decomposition import PCA
from io import BytesIO
import tempfile
import os
import rasterio

# Function to read GeoTIFF images
def read_geotiff(file_like):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(file_like.read())
            tmp_file_path = tmp_file.name

        with rasterio.open(tmp_file_path) as dataset:
            image = dataset.read()
            image = np.moveaxis(image, 0, -1)  # Move channels to the last dimension

        os.remove(tmp_file_path)  # Clean up the temporary file
        return image
    except Exception as e:
        st.error(f"Error reading GeoTIFF: {e}")
        return None

# Function to process the image based on user commands
def process_image(image, commands):
    processed_image = image.copy()
    for command in commands:
        cmd, *param = command.split()
        processed_image = apply_command(processed_image, cmd, param)
    return processed_image

# Function to apply individual image processing commands
def apply_command(image, command, param=None):
    if command == "grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif command == "invert":
        return 255 - image
    elif command == "blur":
        strength = int(param[0]) if param else 5
        return cv2.blur(image, (strength, strength))
    elif command == "resize":
        width, height = (int(param[0]), int(param[1])) if param and len(param) == 2 else (image.shape[1], image.shape[0])
        return cv2.resize(image, (width, height))
    elif command == "crop":
        top, left, bottom, right = (int(param[0]), int(param[1]), int(param[2]), int(param[3])) if param and len(param) == 4 else (0, 0, image.shape[0], image.shape[1])
        return image[top:bottom, left:right]
    elif command == "rotate":
        angle = float(param[0]) if param else 0
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
    elif command in ["add", "subtract", "multiply", "divide"]:
        value = float(param[0]) if param else 0
        if command == "add":
            return cv2.add(image, np.full_like(image, value, dtype=image.dtype))
        elif command == "subtract":
            return cv2.subtract(image, np.full_like(image, value, dtype=image.dtype))
        elif command == "multiply":
            return cv2.multiply(image, np.full_like(image, value, dtype=image.dtype))
        elif command == "divide":
            if value == 0:
                st.error("Division by zero is not allowed.")
                return image
            return cv2.divide(image.astype(np.float32), np.full_like(image, value, dtype=np.float32))
    elif command == "histogram":
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    elif command == "enhance":
        return enhance_image(image)
    elif command == "denoise":
        sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))
        denoised_image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3, channel_axis=-1)
        return (denoised_image * 255).astype(np.uint8)
    elif command == "pca":
        components = int(param[0]) if param else 3
        original_shape = image.shape
        reshaped_image = image.reshape((-1, original_shape[-1]))
        pca = PCA(n_components=components)
        pca_result = pca.fit_transform(reshaped_image)
        pca_image = pca_result.reshape(original_shape)
        return pca_image
    else:
        st.error(f"Invalid command: {command}")
        return image

# Function for enhancing image using histogram equalization
def enhance_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)
    return cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

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
        feature_image = image.copy()
        feature_image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
    else:
        st.error(f"Unsupported feature extraction method: {method}. Supported methods: sift, orb, harris")
        return image

    return feature_image


def main():
    st.title("Image Processing Chat Bot")

    uploaded_file = st.file_uploader("Choose an image file", type=["tif", "tiff", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is not None:
            st.image(image, caption='Uploaded Image', use_column_width=True)

            commands_text = st.text_input("Enter commands (comma-separated)")
            if st.button("Process"):
                commands = commands_text.split(',')
                processed_image = process_image(image, commands)
                st.image(processed_image, caption='Processed Image', use_column_width=True)

                st.header("Additional Operations:")
                while True:
                    additional_command = st.text_input("Enter additional command (or leave blank to finish)")
                    if not additional_command:
                        break
                    processed_image = apply_command(processed_image, *additional_command.split())
                    st.image(processed_image, caption='Processed Image', use_column_width=True)

                if st.button("Download Processed Image"):
                    img_bytes = cv2.imencode(".jpg", processed_image)[1].tobytes()
                    st.download_button(label="Download Processed Image", data=img_bytes, file_name="processed_image.jpg", mime="image/jpeg")

if __name__ == "__main__":
    main()
