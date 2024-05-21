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



app = Flask(__name__)
CORS(app)

# Load the trained model
modelGLCM = pickle.load(open('./models/glcm_dtree.pkl', 'rb'))
modelCNN = pickle.load(open('./models/vgg.pkl', 'rb'))
modelHybrid = pickle.load(open('./models/hybrid_knn.pkl', 'rb'))
modelOneClassSVM = pickle.load(open('./models/one_class_svm.pkl', 'rb'))


# 0 - Dys
# 1 - Non-Dys
# Define a route to accept POST requests
@app.route('/predict', methods=['POST'])
def predict():

    # Retrieve the named parameter from the request
    modelType = request.args.get('model')

    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image']

    if modelType=='GLCM':
        return funGLCM(image)
    elif modelType=='ViT':
        return funViT(image)
    elif modelType=='CNN':
        return funCNN(image)
    elif modelType=='OneClassSVM':
        return funOneClassSVM(image)
    elif modelType=='Hybrid':
        return funHybrid(image)
    else:
        return funEnsemble(image)
    # elif modelType=='Ensemble':

########################################################
# GLCM main - done
def funGLCM(image):
    image = Image.open(image)  # Open the image using PIL
    image = np.array(image)    # Convert PIL image to NumPy array
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to desired dimensions
    resized_image = cv2.resize(gray_image, (512, 512))
    # Reshape the image to match the expected input shape of the model
    data = np.array(resized_image)
    input_data = data.reshape(1, 512, 512)
    # Perform feature extraction on the image
    features = extract_features_for_GLCM(input_data)
    # Make predictions using the loaded model
    predictions = modelGLCM.predict(features)
    # Return the predictions as the API response
    return jsonify({'predictions': predictions.tolist()[0]})
# GLCM helper
def extract_features_for_GLCM(images):
    image_dataset = pd.DataFrame()
    for image in images:
        df = pd.DataFrame()

        dists = [[1],[3],[5],[3],[3]]
        angles = [[0],[0],[0],[np.pi/4],[np.pi/2]]

        for n ,(dist, angle) in enumerate(zip(dists, angles)):

            GLCM = graycomatrix(image, dist, angle)
            GLCM_Energy = graycoprops(GLCM, 'energy')[0]
            df['Energy'+str(n)] = GLCM_Energy
            GLCM_corr = graycoprops(GLCM, 'correlation')[0]
            df['Corr'+str(n)] = GLCM_corr
            GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
            df['Diss_sim'+str(n)] = GLCM_diss
            GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
            df['Homogen'+str(n)] = GLCM_hom
            GLCM_contr = graycoprops(GLCM, 'contrast')[0]
            df['Contrast'+str(n)] = GLCM_contr

        image_dataset = pd.concat([image_dataset, df], ignore_index=True)

    return image_dataset
#######################################################

# ViT
def funViT(image):
    return jsonify({'predictions': "ViT"})


# CNN starting - done
def funCNN(image):
    image = Image.open(image)  # Open the image using PIL
    image = np.array(image)    # Convert PIL image to NumPy array
    img2 = cv2.resize(image, (224,224))
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    img3=img2
    data = np.array(img3)
    data = data/255
    im1 = data.reshape(1,224,224,3)

    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
    for layer in VGG_model.layers:
        layer.trainable = False

    feature_extractor=VGG_model.predict(im1)
    features_1 = feature_extractor.reshape(feature_extractor.shape[0], -1)
    predictions = modelCNN.predict(features_1)
    return jsonify({'predictions': predictions.tolist()[0]})
    # return jsonify({'predictions': "CNN"})



# One Class SVM - done
def funOneClassSVM(image):
    image = Image.open(image)  # Open the image using PIL
    image = np.array(image)    # Convert PIL image to NumPy array
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (512,512))
    img=img1
    data = np.array(img1)
    im = data.reshape(1,512,512)
    # feature extractor for GLCM XGB is same as GLCM
    train_extr = extract_features_for_GLCM(im)
    predictions=modelOneClassSVM.predict(train_extr)
    return jsonify({'predictions': predictions.tolist()[0]})
    # print('predicted label',l)
    # return jsonify({'predictions': "One Class SVM"})



########################################################
# Hybrid model - done
def funHybrid(image):
    image = Image.open(image)  # Open the image using PIL
    image = np.array(image)    # Convert PIL image to NumPy array
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img1, (3,3), 0)
    img1 = cv2.Canny(img1, 100, 200)
    img1 = cv2.resize(img1, (736,736))
    data = np.array(img1)
    im = data.reshape(1,736,736)
    train_extr = feature_extractor_for_hybrid(im)
    predictions=modelHybrid.predict(train_extr)
    return jsonify({'predictions': predictions.tolist()[0]})
    # return jsonify({'predictions': "Hybrid"})
# Hybrid Helper
def feature_extractor_for_hybrid(images):
    image_dataset = pd.DataFrame()
    i = 0
    for image in images:
        print("\nProcessing Image : ",i)
        i += 1
        df = pd.DataFrame()
        dists = [[1],[3],[5],[3],[3]]
        angles = [[0],[0],[0],[np.pi/4],[np.pi/2]]

        for n ,(dist, angle) in enumerate(zip(dists, angles)):

            GLCM = graycomatrix(image, dist, angle)
            GLCM_Energy = graycoprops(GLCM, 'energy')[0]
            df['Energy'+str(n)] = GLCM_Energy
            GLCM_corr = graycoprops(GLCM, 'correlation')[0]
            df['Corr'+str(n)] = GLCM_corr
            GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
            df['Diss_sim'+str(n)] = GLCM_diss
            GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
            df['Homogen'+str(n)] = GLCM_hom
            GLCM_contr = graycoprops(GLCM, 'contrast')[0]
            df['Contrast'+str(n)] = GLCM_contr

        df['pressure'] = [hp.find_pressure(image)]
        h=hp.find_lineSpace_letterSize_relativeLineSpace(image)
        df['linespace']=[h[0]]
        df['letterSize']=[h[1]]
        df['relativeLineSpace']=[h[2]]
        df['Baseline_angle'] = [hp.find_baseline_angle(image)]
        df['word_spacing'] = [hp.find_word_spacing(image)]
        df['Slant_angle'] = [hp.find_slant_angle(image)]
        df.fillna(0,inplace=True)
        # image_dataset = image_dataset.append(df)
        image_dataset = pd.concat([image_dataset, df], ignore_index=True)

    scaler = MinMaxScaler()
    four=scaler.fit_transform(image_dataset)
    data=pd.DataFrame(four)
    return data
########################################################

# Ensemble model
def funEnsemble(image):
    glcm=cnn=hybrid=oneClassSVM=vit=0
    dysgraphia=non_dysgraphia=0

    temp = funGLCM(image).json
    glcm = temp['predictions']
    
    temp = funCNN(image).json
    cnn = temp['predictions']

    temp = funHybrid(image).json
    hybrid = temp['predictions']

    temp = funOneClassSVM(image).json
    oneClassSVM = temp['predictions']

    temp = funViT(image).json
    # vit = temp['predictions']
    vit = 1


    if glcm==0 : dysgraphia = dysgraphia+1 
    else: non_dysgraphia=non_dysgraphia+1

    if cnn==0 : dysgraphia = dysgraphia+1 
    else: non_dysgraphia=non_dysgraphia+1

    if hybrid==0 : dysgraphia = dysgraphia+1 
    else: non_dysgraphia=non_dysgraphia+1

    if oneClassSVM==0 : dysgraphia = dysgraphia+1 
    else: non_dysgraphia=non_dysgraphia+1

    if vit==0 : dysgraphia = dysgraphia+1 
    else: non_dysgraphia=non_dysgraphia+1

    print(dysgraphia,non_dysgraphia)
    if dysgraphia>non_dysgraphia:
        return jsonify({'predictions': 0})
    else:
        return jsonify({'predictions': 1})
    # return jsonify({'predictions': "Ensemble"})


# starting point
@app.route('/')
def hello_world():
    return 'Dysgraphia prediction mode. Please hit the POST request along with image as a payload'

if __name__ == '__main__':
    app.run(port=7860)


