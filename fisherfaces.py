import os
import cv2
import glob
import random
import numpy as np
emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprize"]
fishface = cv2.face.FisherFaceRecognizer_create()
data = {}
def get_files(emotion):
    training = glob.glob("../dataset/" %emotion) #training path
    prediction=glob.glob("../dataset/" %emotion) #prediction path
    return training, prediction
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels
def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print("training fisher face classifier")
    print("size of training set:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    print("predicting")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100*correct)/(correct + incorrect))
    
metascore = []
for i in range(0,10):
    correct = run_recognizer()
    print(correct, " -> percent correct!")
    metascore.append(correct)
print("\n\nend score:", np.mean(metascore), "percent correct!")