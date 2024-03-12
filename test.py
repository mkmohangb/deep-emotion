from fer_dataset import FerDataset
from model import DeepEmotion
import pandas as pd
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(args):
    df = pd.read_csv("icml_face_data.csv")
    test_dataset = FerDataset(df[df[' Usage'] == 'PrivateTest'])
    batch_size = 64
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=batch_size, 
                            shuffle=False, drop_last=True)
    print(f"number of images in test set is {len(test_dataset)}")
    net = DeepEmotion()
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.to(device)
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()

    test(args)

