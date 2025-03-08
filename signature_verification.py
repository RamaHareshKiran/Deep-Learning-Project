import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Paths
MODEL_DIR = "models"
USER_DATA_DIR = "user_data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Step 1: Load Pretrained Models
def build_feature_extractor(model_name):
    if model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    elif model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
    else:
        raise ValueError("Model not supported.")
    
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

feature_extractor_resnet = build_feature_extractor('resnet50')
feature_extractor_vgg16 = build_feature_extractor('vgg16')

# Step 2: Image Preprocessing
def preprocess_image(img_path, model_type):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_type == 'resnet50':
        return tf.keras.applications.resnet50.preprocess_input(img_array)
    else:
        return tf.keras.applications.vgg16.preprocess_input(img_array)

# Step 3: Store User Signature Features
def store_signature_features(user_id, img_path):
    user_model_path = os.path.join(MODEL_DIR, f"{user_id}.pkl")

    img_resnet = preprocess_image(img_path, 'resnet50')
    img_vgg16 = preprocess_image(img_path, 'vgg16')

    features_resnet = feature_extractor_resnet.predict(img_resnet).reshape(1, -1)
    features_vgg16 = feature_extractor_vgg16.predict(img_vgg16).reshape(1, -1)

    if os.path.exists(user_model_path):
        with open(user_model_path, "rb") as f:
            user_signatures = pickle.load(f)
    else:
        user_signatures = []

    user_signatures.append((features_resnet, features_vgg16))

    with open(user_model_path, "wb") as f:
        pickle.dump(user_signatures, f)

    print(f"✅ Stored signature for {user_id}. Total Signatures: {len(user_signatures)}")

# Step 4: Verify Signature
def verify_signature(user_id, input_img_path, threshold=0.6):
    user_model_path = os.path.join(MODEL_DIR, f"{user_id}.pkl")

    if not os.path.exists(user_model_path):
        return "❌ User not found!"

    with open(user_model_path, "rb") as f:
        stored_signatures = pickle.load(f)

    input_resnet = preprocess_image(input_img_path, 'resnet50')
    input_vgg16 = preprocess_image(input_img_path, 'vgg16')

    input_features_resnet = feature_extractor_resnet.predict(input_resnet).reshape(1, -1)
    input_features_vgg16 = feature_extractor_vgg16.predict(input_vgg16).reshape(1, -1)

    resnet_scores = [cosine_similarity(stored[0], input_features_resnet)[0][0] for stored in stored_signatures]
    vgg_scores = [cosine_similarity(stored[1], input_features_vgg16)[0][0] for stored in stored_signatures]

    avg_similarity_resnet = np.mean(resnet_scores)
    avg_similarity_vgg = np.mean(vgg_scores)

    resnet_result = "Genuine" if avg_similarity_resnet >= threshold else "Forged"
    vgg_result = "Genuine" if avg_similarity_vgg >= threshold else "Forged"

    return f"ResNet: {resnet_result} (Score: {avg_similarity_resnet:.2f}), VGG: {vgg_result} (Score: {avg_similarity_vgg:.2f})"
