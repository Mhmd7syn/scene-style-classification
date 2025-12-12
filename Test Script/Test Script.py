import os
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import Script_Helpers, CLIP_VIT_Implementation

# Sets the policy to mixed_float16 for faster Computation
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# ================= CONFIGURATION =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '../Models'))
TEST_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../Dataset/StyleClassificationIndoors/temp_test'))

BATCH_SIZE = 32
IMG_SIZE = (300, 300)
CLASS_NAMES = ["asian", "boho", "coastal", "contemporary", "craftsman", "eclectic", "farmhouse", "french-country", "industrial", "mediterranean", "minimalist", "modern", "scandinavian", "shabby-chic-style", "southwestern", "tropical", "victorian"]

MODEL_FILENAMES = [
    "custom_vit_L14_model.keras",
    "custom_vit_patch32_model.keras",
    "custom_resnet_model.keras",
    "custom_efficientnetB0_model.keras",
    "custom_efficientnetB3_model.keras",
    "custom_inception_model.keras"
]

# ================= MAIN EXECUTION =================
CUSTOM_OBJECTS = {
    'RandomSaturation': Script_Helpers.RandomSaturation,
    'PlanckianJitter': Script_Helpers.PlanckianJitter,
    'CLIPNormalization': Script_Helpers.CLIPNormalization,
    'caffe_preprocess': Script_Helpers.caffe_preprocess,
    'QuickGELU': CLIP_VIT_Implementation.QuickGELU,
    'CLIPAttention': CLIP_VIT_Implementation.CLIPAttention,
    'CLIPMLP': CLIP_VIT_Implementation.CLIPMLP,
    'CLIPEncoderLayer': CLIP_VIT_Implementation.CLIPEncoderLayer,
    'swish': tf.keras.activations.swish,
}

# 1. Prepare Data
print(f"Scanning files in {TEST_DATA_DIR}...")
all_paths, true_labels_str, found_classes, is_labeled = Script_Helpers.get_data_info(TEST_DATA_DIR)

# Resolve Class Mapping
if is_labeled:
    print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    try:
        y_true = [class_to_idx[label] for label in true_labels_str]
    except KeyError as e:
        print(f"\nERROR: Found a test folder {e} that is NOT in your CLASS_NAMES list.")
        exit()

# 2. Build Dataset
if len(all_paths) == 0:
    print("No images found. Check directory paths.")
    exit()

test_ds = tf.data.Dataset.from_tensor_slices(all_paths)
test_ds = test_ds.map(Script_Helpers.load_image_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"Pipeline created. Found {len(all_paths)} images.")

# 3. Model Loop
results_summary = []

for filename in MODEL_FILENAMES:
    # EDIT: Join with MODELS_DIR instead of BASE_DIR
    model_path = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(model_path):
        print(f"\nSkipping: File not found -> {model_path}")
        continue
    
    print(f"\n" + "="*40)
    print(f"Processing: {filename}")
    print("="*40)
    
    try:
        # Load Model
        current_custom_objects = CUSTOM_OBJECTS.copy()
        
        if 'vit' in filename.lower():
            if "patch32" in filename.lower():
                print(" >> Mode: Patch32 (Small - 768)")
                TransformerClass = CLIP_VIT_Implementation.get_clip_transformer_class(
                    default_hidden=768, default_patch=32, default_heads=12, default_layers=12, default_inter=3072
                )
            else:
                print(" >> Mode: L14 (Large - 1024)")
                TransformerClass = CLIP_VIT_Implementation.get_clip_transformer_class(
                    default_hidden=1024, default_patch=14, default_heads=16, default_layers=24, default_inter=4096
                )

            current_custom_objects['CLIPVisionTransformer'] = TransformerClass
       
        model = tf.keras.models.load_model(model_path, custom_objects=current_custom_objects)
        
        # Predict
        print("Predicting...")
        predictions = model.predict(test_ds, verbose=1)
        class_logits = predictions[0] if isinstance(predictions, list) else predictions
        predicted_ids = np.argmax(class_logits, axis=1)

        # Calculate Accuracy if labeled
        acc_text = "N/A"
        if is_labeled:
            acc = accuracy_score(y_true, predicted_ids)
            acc_text = f"{acc:.4f}"
            print(f"\n>>> ACCURACY: {acc:.2%}")
            
        results_summary.append({'Model': filename, 'Accuracy': acc_text})

        # Save CSV
        print("Saving submission file...")
        test_file_names = [os.path.basename(path) for path in all_paths]
        
        df = pd.DataFrame({
            'ImageName': test_file_names,
            'ClassLabel': predicted_ids
        })
        clean_name = filename.replace('.keras', '').replace('.h5', '')
        output_csv = os.path.join(BASE_DIR+"/Submissions", f'submission_{clean_name}.csv')
        df.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
        import traceback
        traceback.print_exc()

# 4. Final Summary
if is_labeled:
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(pd.DataFrame(results_summary))