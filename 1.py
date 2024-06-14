import cv2
import numpy as np
import streamlit as st
from skimage import exposure
from skimage.restoration import denoise_nl_means, estimate_sigma
from sklearn.decomposition import PCA
from io import BytesIO
from osgeo import gdal
import tempfile
import os

# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.set_option('server.maxUploadSize', 1024)  # 1 GB limit

# Function to capture and recognize speech (with follow-up prompt and retry)
def get_voice_command(retries=3):
    # Placeholder for offline speech recognition setup
    st.warning("Offline speech recognition not implemented.")
    return None

# Function to read GeoTIFF images
def read_geotiff(file_like):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(file_like.read())
            tmp_file_path = tmp_file.name

        dataset = gdal.Open(tmp_file_path)
        image = dataset.ReadAsArray()
        image = np.moveaxis(image, 0, -1)  # Move channels to the last dimension

        os.remove(tmp_file_path)  # Clean up the temporary file
        return image
    except Exception as e:
        st.error(f"Error reading GeoTIFF: {e}")
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
    elif command == "enhance":
        enhanced_image = enhance_image(image)
        return enhanced_image
    elif command == "denoise":
        sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))
        denoised_image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3, channel_axis=-1)
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
    elif command == "feature_extraction":
        method = param[0] if param else "sift"
        return feature_extraction(image, method)
    else:
        st.error("Invalid command. Supported commands: grayscale, invert, blur [strength], resize [width height], crop [top left bottom right], rotate [angle], add [value], subtract [value], multiply [value], divide [value], histogram, enhance, denoise, pca [components], sift, mean, median, max, min, feature_extraction [method]")
        return image

# # Function for enhancing image using histogram equalization
# def enhance_image(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     enhanced_image = exposure.equalize_hist(gray_image)
#     enhanced_image = (enhanced_image * 255).astype(np.uint8)
#     return enhanced_image

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

# Function to read GeoTIFF images
# Function to read GeoTIFF images
def read_geotiff(file_like):
    try:
        # Create a temporary file
        fd, tmp_file_path = tempfile.mkstemp(suffix='.tif')
        
        # Write the file content
        with open(tmp_file_path, 'wb') as tmp_file:
            tmp_file.write(file_like.read())

        # Close the file descriptor
        os.close(fd)

        # Read the GeoTIFF using GDAL
        gdal_dataset = gdal.Open(tmp_file_path)
        image = gdal_dataset.ReadAsArray()
        image = np.moveaxis(image, 0, -1)  # Move channels to the last dimension

        # Clean up the temporary file
        gdal_dataset = None  # Close the dataset
        os.remove(tmp_file_path)

        return image
    except Exception as e:
        st.error(f"Error reading GeoTIFF: {e}")
        return None
# Function for enhancing image using histogram equalization
def enhance_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = exposure.equalize_hist(gray_image)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    return enhanced_image

def download_image(image, uploaded_file_name):
    ext = uploaded_file_name.split('.')[-1]
    img_bytes = cv2.imencode(f".{ext}", image)[1].tobytes()
    st.download_button(label="Download Processed Image", data=img_bytes, file_name=f"processed_image.{ext}", mime=f"image/{ext}")

# Main function for Streamlit app
def main():
    st.title("Image Processing App")

    uploaded_file = st.file_uploader("Choose an image file", type=["tif", "tiff", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name
        if uploaded_file_name.endswith((".tif", ".tiff")):
            file_like = BytesIO(uploaded_file.read())  # Create a file-like object
            image = read_geotiff(file_like)
        else:
            image_data = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is not None:
            st.image(image, caption='Uploaded Image', use_column_width=True)

            use_voice_command = st.checkbox("Use voice command (Not available offline)")

            if use_voice_command:
                st.warning("Offline speech recognition not implemented.")
                command = None
            else:
                command_text = st.text_input("Enter command")
                command = command_text.lower().split() if command_text else None

            if command:
                processed_image = process_image(image.copy(), command[0], command[1:])
                if processed_image is not None:
                    st.image(processed_image, caption='Processed Image', use_column_width=True)
                    download_image(processed_image, uploaded_file_name)  # Add download button

if __name__ == "__main__":
    main()