import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import glob
import re
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# Sets the policy to mixed_float16 for faster Computation
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

IMG_SIZE = (300, 300)

# ================= Helper Functions & Classes =================
def natural_key(text):
    """Sorts strings that contain numbers in a human-friendly way."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def load_image_with_pil_logic(path_tensor):
    path = path_tensor.numpy().decode('utf-8')
    try:
        with Image.open(path) as img:
            img = img.convert('RGB')
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            return np.array(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        # Return black image on failure to keep batch size consistent
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)

def load_image_wrapper(path):
    img = tf.py_function(func=load_image_with_pil_logic, inp=[path], Tout=tf.uint8)
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img

def get_data_info(data_dir):
    """Scans directory, returns sorted paths and determines if data is labeled."""
    search_pattern = os.path.join(data_dir, "**/*.*")
    all_paths = glob.glob(search_pattern, recursive=True)
    
    # Filter out non-image files just in case
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    all_paths = [p for p in all_paths if os.path.splitext(p)[1].lower() in valid_exts]
    all_paths.sort(key=natural_key)

    if not all_paths:
        # Returns empty lists instead of None to prevent unpacking errors
        print("No images found! Check your TEST_DATA_DIR path.")
        return [], [], None, False

    # Check for labels
    first_parent = os.path.basename(os.path.dirname(all_paths[0]))
    root_folder_name = os.path.basename(data_dir)
    
    is_labeled = first_parent != root_folder_name and first_parent != "test"
    
    true_labels = []
    found_classes = set()

    if is_labeled:
        print("Structure detected: Labeled subdirectories.")
        for p in all_paths:
            folder_name = os.path.basename(os.path.dirname(p))
            true_labels.append(folder_name)
            found_classes.add(folder_name)
    else:
        print("Structure detected: Flat directory (Unlabeled).")

    return all_paths, true_labels, found_classes, is_labeled

@tf.keras.utils.register_keras_serializable()
class RandomSaturation(layers.Layer):
    def __init__(self, lower, upper, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, images, training=True):
        if not training:
            return images
        return tf.image.random_saturation(images, self.lower, self.upper)
    
    def get_config(self):
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper})
        return config

@tf.keras.utils.register_keras_serializable()
class PlanckianJitter(layers.Layer):
    def __init__(self, factor=0.1, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, images, training=True):
        if not training:
            return images

        images = tf.cast(images, tf.float32)
        batch_size = tf.shape(images)[0]
        delta = tf.random.uniform([batch_size, 1, 1, 1], 
                                  minval=-self.factor, 
                                  maxval=self.factor)
        r_gain = 1.0 + delta
        g_gain = 1.0 
        b_gain = 1.0 - delta
        gains = tf.concat([r_gain, g_gain * tf.ones_like(r_gain), b_gain], axis=-1)
        
        return tf.clip_by_value(images * gains, 0.0, 255.0)

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config

@tf.keras.utils.register_keras_serializable()
class CLIPNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.constant([0.48145466, 0.4578275, 0.40821073], dtype=tf.float32)
        self.std = tf.constant([0.26862954, 0.26130258, 0.27577711], dtype=tf.float32)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        return (x - self.mean) / self.std
    
    def get_config(self):
        return super().get_config()

def caffe_preprocess(images):
        x = tf.cast(images, tf.float32)
        x = x[..., ::-1] # RGB -> BGR
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        x = x - mean
        return x