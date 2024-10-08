import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load your trained model
model1 = tf.keras.models.load_model('./unet_model/unet_100_epochs.h5')
model2 = tf.keras.models.load_model('./unet_model/unet_100_epochs.h5')  

# Define constants for image size
IMG_HEIGHT = 256
IMG_WIDTH = 256

def process_image(image,model):
    """Preprocess the image for the model and make predictions."""
    # Convert the uploaded PIL image to a NumPy array
    img_array = np.array(image)

    # Get the original dimensions of the image
    orig_height, orig_width = img_array.shape[:2]

    # If the image is smaller than 256x256, resize it to 256x256
    if orig_height < IMG_HEIGHT or orig_width < IMG_WIDTH:
        img_resized = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_batch = np.expand_dims(img_rgb, axis=0)

        # Make prediction for the entire image
        pred = model.predict(img_batch)
        pred_image = np.argmax(pred[0], axis=-1).astype(np.uint8)

        # Resize prediction back to original size
        return cv2.resize(pred_image, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

    # Split the image into patches if it's larger
    patches = []
    for i in range(0, orig_height, IMG_HEIGHT):
        for j in range(0, orig_width, IMG_WIDTH):
            patch = img_array[i:i+IMG_HEIGHT, j:j+IMG_WIDTH]
            # If the patch is smaller than 256x256, pad it with zeros
            if patch.shape[0] < IMG_HEIGHT or patch.shape[1] < IMG_WIDTH:
                padded_patch = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch
            patches.append(patch)

    # Predict for each patch
    predictions = []
    for patch in patches:
        img_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        img_batch = np.expand_dims(img_rgb, axis=0)
        pred = model.predict(img_batch)
        pred_image = np.argmax(pred[0], axis=-1).astype(np.uint8)
        predictions.append(pred_image)

    # Merge the predicted patches back together
    merged_pred = np.zeros((orig_height, orig_width), dtype=np.uint8)
    patch_idx = 0
    for i in range(0, orig_height, IMG_HEIGHT):
        for j in range(0, orig_width, IMG_WIDTH):
            patch_pred = predictions[patch_idx]
            patch_height, patch_width = min(IMG_HEIGHT, orig_height - i), min(IMG_WIDTH, orig_width - j)
            merged_pred[i:i+patch_height, j:j+patch_width] = patch_pred[:patch_height, :patch_width]
            patch_idx += 1

    return merged_pred

# Streamlit app
def main():
    st.title("Image Segmentation with Two Models")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image when the button is clicked
        if st.button("Process Image with Both Models"):
            with st.spinner("Processing..."):
                # Process the image using the first model
                predicted_img1 = process_image(image, model1)
                # Process the image using the second model
                predicted_img2 = process_image(image, model2)

                # Set up the grid for displaying side-by-side results
                gs = GridSpec(1, 2)  # Create a grid for 2 plots (1 row, 2 columns)
                fig = plt.figure(dpi=200)  # Set the figure resolution

                # Display the first model's predicted segmentation
                ax1 = fig.add_subplot(gs[0])
                ax1.imshow(predicted_img1, cmap='gray')
                ax1.axis('off')
                ax1.set_title("UNet Prediction")

                # Display the second model's predicted segmentation
                ax2 = fig.add_subplot(gs[1])
                ax2.imshow(predicted_img2, cmap='gray')
                ax2.axis('off')
                ax2.set_title("UNet++ Prediction")

                # Adjust layout to prevent overlapping
                plt.tight_layout()

                # Show the complete plot
                st.pyplot(fig)

if __name__ == "__main__":
    main()