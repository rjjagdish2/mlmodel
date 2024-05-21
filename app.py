from flask_cors import CORS
from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import cv2
from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import MinMaxScaler
import helper as hp
from huggingface_hub import login, HfApi

# Log in to Hugging Face
api = HfApi()
api.login(token="YOUR_HUGGING_FACE_TOKEN")

app = Flask(_name_)
CORS(app)

# Load the trained models
modelGLCM = pickle.load(open('./models/glcm_dtree.pkl', 'rb'))
modelCNN = pickle.load(open('./models/vgg.pkl', 'rb'))
modelHybrid = pickle.load(open('./models/hybrid_knn.pkl', 'rb'))
modelOneClassSVM = pickle.load(open('./models/one_class_svm.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    modelType = request.args.get('model')
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image']
    if modelType == 'GLCM':
        return funGLCM(image)
    elif modelType == 'ViT':
        return funViT(image)
    elif modelType == 'CNN':
        return funCNN(image)
    elif modelType == 'OneClassSVM':
        return funOneClassSVM(image)
    elif modelType == 'Hybrid':
        return funHybrid(image)
    else:
        return funEnsemble(image)

def funGLCM(image):
    image = Image.open(image)
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (512, 512))
    data = np.array(resized_image).reshape(1, 512, 512)
    features = extract_features_for_GLCM(data)
    predictions = modelGLCM.predict(features)
    return jsonify({'predictions': predictions.tolist()[0]})

def extract_features_for_GLCM(images):
    image_dataset = pd.DataFrame()
    for image in images:
        df = pd.DataFrame()
        dists = [[1], [3], [5], [3], [3]]
        angles = [[0], [0], [0], [np.pi / 4], [np.pi / 2]]
        for n, (dist, angle) in enumerate(zip(dists, angles)):
            GLCM = graycomatrix(image, dist, angle)
            GLCM_Energy = graycoprops(GLCM, 'energy')[0]
            df['Energy' + str(n)] = GLCM_Energy
            GLCM_corr = graycoprops(GLCM, 'correlation')[0]
            df['Corr' + str(n)] = GLCM_corr
            GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
            df['Diss_sim' + str(n)] = GLCM_diss
            GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
            df['Homogen' + str(n)] = GLCM_hom
            GLCM_contr = graycoprops(GLCM, 'contrast')[0]
            df['Contrast' + str(n)] = GLCM_contr
        image_dataset = pd.concat([image_dataset, df], ignore_index=True)
    return image_dataset

def funViT(image):
    return jsonify({'predictions': "ViT"})

def funCNN(image):
    image = Image.open(image)
    image = np.array(image)
    img2 = cv2.resize(image, (224, 224))
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    img3 = img2
    data = np.array(img3) / 255
    im1 = data.reshape(1, 224, 224, 3)
    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in VGG_model.layers:
        layer.trainable = False
    feature_extractor = VGG_model.predict(im1)
    features_1 = feature_extractor.reshape(feature_extractor.shape[0], -1)
    predictions = modelCNN.predict(features_1)
    return jsonify({'predictions': predictions.tolist()[0]})

def funOneClassSVM(image):
    image = Image.open(image)
    image = np.array(image)
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (512, 512))
    data = np.array(img1).reshape(1, 512, 512)
    train_extr = extract_features_for_GLCM(data)
    predictions = modelOneClassSVM.predict(train_extr)
    return jsonify({'predictions': predictions.tolist()[0]})

def funHybrid(image):
    image = Image.open(image)
    image = np.array(image)
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img1, (3, 3), 0)
    img1 = cv2.Canny(img1, 100, 200)
    img1 = cv2.resize(img1, (736, 736))
    data = np.array(img1).reshape(1, 736, 736)
    train_extr = feature_extractor_for_hybrid(data)
    predictions = modelHybrid.predict(train_extr)
    return jsonify({'predictions': predictions.tolist()[0]})

def feature_extractor_for_hybrid(images):
    image_dataset = pd.DataFrame()
    for image in images:
        df = pd.DataFrame()
        dists = [[1], [3], [5], [3], [3]]
        angles = [[0], [0], [0], [np.pi / 4], [np.pi / 2]]
        for n, (dist, angle) in enumerate(zip(dists, angles)):
            GLCM = graycomatrix(image, dist, angle)
            GLCM_Energy = graycoprops(GLCM, 'energy')[0]
            df['Energy' + str(n)] = GLCM_Energy
            GLCM_corr = graycoprops(GLCM, 'correlation')[0]
            df['Corr' + str(n)] = GLCM_corr
            GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
            df['Diss_sim' + str(n)] = GLCM_diss
            GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
            df['Homogen' + str(n)] = GLCM_hom
            GLCM_contr = graycoprops(GLCM, 'contrast')[0]
            df['Contrast' + str(n)] = GLCM_contr
        df['pressure'] = [hp.find_pressure(image)]
        h = hp.find_lineSpace_letterSize_relativeLineSpace(image)
        df['linespace'] = [h[0]]
        df['letterSize'] = [h[1]]
        df['relativeLineSpace'] = [h[2]]
        df['Baseline_angle'] = [hp.find_baseline_angle(image)]
        df['word_spacing'] = [hp.find_word_spacing(image)]
        df['Slant_angle'] = [hp.find_slant_angle(image)]
        df.fillna(0, inplace=True)
        image_dataset = pd.concat([image_dataset, df], ignore_index=True)
    scaler = MinMaxScaler()
    four = scaler.fit_transform(image_dataset)
    return pd.DataFrame(four)

def funEnsemble(image):
    glcm = cnn = hybrid = oneClassSVM = vit = 0
    dysgraphia = non_dysgraphia = 0

    temp = funGLCM(image).json
    glcm = temp['predictions']

    temp = funCNN(image).json
    cnn = temp['predictions']

    temp = funHybrid(image).json
    hybrid = temp['predictions']

    temp = funOneClassSVM(image).json
    oneClassSVM = temp['predictions']

    temp = funViT(image).json
    vit = 1

    if glcm == 0: dysgraphia += 1 
    else: non_dysgraphia += 1

    if cnn == 0: dysgraphia += 1 
    else: non_dysgraphia += 1

    if hybrid == 0: dysgraphia += 1 
    else: non_dysgraphia += 1

    if oneClassSVM == 0: dysgraphia += 1 
    else: non_dysgraphia += 1

    if vit == 0: dysgraphia += 1 
    else: non_dysgraphia += 1

    return jsonify({'predictions': 0 if dysgraphia > non_dysgraphia else 1})

@app.route('/')
def hello_world():
    return 'Dysgraphia prediction model. Please hit the POST request along with the image as payload'

if _name_ == '_main_':
    app.run(port=7860)
