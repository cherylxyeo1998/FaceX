# convert image file into 128-point vector, using facex api
import requests, json, os
import numpy as np

Headers = {'user_id':'5def6cea6ea12f126fdd4ce1'}

IMAGE_PATH = './lfw/Adam/001.jpg'
IMAGE_PATH2 = './lfw/Adam/002.jpg'
API_URL = "http://facexapi.com/get_face_vec?face_det=1"
Files = {'img': open(IMAGE_PATH,'rb')}
Files2 = {'img': open(IMAGE_PATH2,'rb')}

def getArray(Files):
    r = requests.post(API_URL,headers=Headers,files=Files)
    imgDict = json.loads(r.text)
    return imgDict['data']['vector']


refAdam = np.array(getArray(Files))
testAdam = np.array(getArray(Files2))
print(refAdam)
print(testAdam)
print('Euclidean distance is {}'.format(np.linalg.norm(refAdam-testAdam)))