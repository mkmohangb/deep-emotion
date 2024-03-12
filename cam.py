import argparse
import cv2
from model import DeepEmotion
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--model')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ('Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral')

net = DeepEmotion()
net.load_state_dict(torch.load(args.model, map_location=device))
net.to(device)
net.eval()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
roi = np.zeros((48, 48), dtype=np.float32)
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi,(48,48))
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


    with torch.no_grad():
        images = np.zeros(shape=(1, 48, 48), dtype=np.float32)
        images[0] = roi / 255.
        outputs = net(torch.tensor(images[None]).to(device))
        _, predicted = torch.max(outputs.data, 1)
        prediction = classes[predicted]

    img = cv2.putText(img, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                      (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('img', img)
    # Stop if (Q) key is pressed
    k = cv2.waitKey(30)
    if k==ord("q"):
        break

cap.release()
