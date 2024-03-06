import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def prepare_data(df):
    img_array = np.zeros(shape=(len(df), 48, 48), dtype=np.float32)
    img_label = np.array(list(map(int, df['emotion'])))

    for i, row in enumerate(df.index):
        img = np.fromstring(df.loc[row, ' pixels'], dtype=int, sep=' ')
        img = np.reshape(img, (48, 48))
        img_array[i] = img

    return img_array, img_label

class FerDataset(Dataset):
    def __init__(self, df):
        self.img_array, self.labels = prepare_data(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.ToTensor()])
        return (transform(self.img_array[idx]), self.labels[idx])

    def get_dist(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(map(emotions.get, unique), counts))
