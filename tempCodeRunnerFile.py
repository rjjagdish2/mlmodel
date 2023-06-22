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