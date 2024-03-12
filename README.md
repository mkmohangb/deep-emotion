#### Implementation of [DeepEmotion](https://www.mdpi.com/1424-8220/21/9/3046)

<img src="model-arch.png" width=600 title="Model architecture">

- Download and prepare [FER 2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- 48x48 pixel grayscale images of faces
- emotion shown in the facial expression in to one of seven categories (0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral)
```
  kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge
  unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip
```
- The file ```icml_face_data.csv``` contains the train(28709 images), validation(3589 images) and test data(3589 images).

- Training and Testing (torch - 2.1.0)
```
  python train.py --log_dir logs -e 100 -b 256
  python test.py --model deep_emotion-256-0.005.pt
```
- Test accuracy of 45% after training for 100 epochs
