import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import base64
from tqdm import tqdm
import os

import PIL.Image
import time
import functools
tf.executing_eagerly()
st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(
    page_title="Style Transfer", layout="wide", page_icon="./images/icon.png"
)

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def load_image(image, image_size=(256, 256), preserve_aspect_ratio=True):
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = plt.imread(image).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

#This function is giving the user the ability to load images

def load_img(path_to_img):
  # max dimension of the images that we are importing
  max_dim = 512
#Here we are loading our file, decoding it into a 3 dimensional tensor and converting it into a file based on format
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def write_image(dg, arr):
    arr = np.uint8(np.clip(arr / 255.0, 0, 1) * 255)
    dg.image(arr, use_column_width=True)
    return dg


def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpg;base64,{img_str}" target="_blank">Download result</a>'
    return href


def pil_to_bytes(model_output):
    pil_image = Image.fromarray(np.squeeze(model_output * 255).astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    byte_image = buffer.getvalue()
    return byte_image


st.sidebar.title("Style Transfer")
st.sidebar.markdown(
    "Neural style transfer is an optimization technique used to take two images:</br>- *Content image* </br>- *Style reference image* (such as an artwork by a famous painter)</br>Blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.",
    unsafe_allow_html=True)
st.sidebar.markdown(
    "[View on Tensorflow.org](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization)")

content_image_buffer = st.sidebar.file_uploader("upload content image", type=["png", "jpeg", "jpg"],
                                                accept_multiple_files=False, key=None, help="content image")
# style_image_buffer = st.sidebar.file_uploader("upload style image", type=["png", "jpeg", "jpg"],
#                                               accept_multiple_files=False, key=None, help="style image")

col1, col2, col3 = st.columns(3)

st.markdown("## Try Style Transfer by uploading content and style pictures from the sidebar :art:")


with st.spinner("Loading content image.."):
    if content_image_buffer:
        col1.header("Content Image")
        col1.image(content_image_buffer, use_column_width=True)
        content_img_size = (500, 500)
        content_image = load_image(content_image_buffer, content_img_size)

if st.sidebar.button(label="Generate"):
    if content_image_buffer:
        with st.spinner('Generating Stylized image ...'):
            hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
            name_list = os.listdir('style_images')
            print(name_list)
            load_folder = 'style_images'
            for name in tqdm(name_list):
                try:
                    load_path = os.path.join(load_folder, name)
                    style_image = load_img(load_path)
                    stylized_image = hub_module(content_image,tf.constant(style_image))[0]
                    #print(stylized_image)
                    op = tensor_to_image(stylized_image)

                    op.save(os.path.join('out/' + "NFT_" + name))
                    st.image(op, width=500)
                    st.download_button(label="Download result", data=pil_to_bytes(stylized_image),
                                       file_name='out/' + "NFT_" + name, mime="image/png")
                except:
                    print("  Error occur :: ")
                    print(name)


    else:
        st.sidebar.markdown("Please chose content and style pictures.")