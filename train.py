from fer_dataset import FerDataset
from model import DeepEmotion
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(args):
    from os import path
    model = DeepEmotion().to(device)
    train_logger, valid_logger = None, None
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    df = pd.read_csv("icml_face_data.csv")
    train_dataset = FerDataset(df[df[' Usage'] == 'Training'])
    val_dataset = FerDataset(df[df[' Usage'] == 'PublicTest'])
    
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, 
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=0, batch_size=args.batch_size, 
                              shuffle=True, drop_last=True)

    cls_weight = (df[df[' Usage']=='Training']['emotion'].value_counts().sort_index() / len(df[df[' Usage']=='Training'])).tolist()

    lr = 0.005
    optimizer = torch.optim.Adam([
        {'params': model.feat_extraction.parameters()},
        {'params': model.fc_de.parameters(), 'weight_decay': 0.001},
        {'params': model.localization.parameters()},
        {'params': model.fc_loc.parameters()},
        ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_obj = torch.nn.CrossEntropyLoss()#weight=torch.tensor(cls_weight))
    best_vloss = 1_000_000.

    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch + 1))
        last_loss = 0.
        running_loss = 0.
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            input, label = data
            output = model(input.to(device))
            loss = loss_obj(output, label.to(device))
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if i % 40 == 39:
                last_loss = running_loss / 40
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(train_loader) + i + 1
                train_logger.add_scalar("train/loss", last_loss, tb_x)
                running_loss = 0

        avg_loss = last_loss
        running_vloss = 0.0

        model.eval()
        with torch.no_grad():
            for i,vdata in enumerate(val_loader): 
                vinputs, vlabels = vdata
                voutputs = model(vinputs.to(device))
                vloss = loss_obj(voutputs, vlabels.to(device))
                running_vloss += vloss
        
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {} lr {}'.format(avg_loss, avg_vloss, optimizer.param_groups[0]['lr']))
        valid_logger.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            print("saving model...as vloss reduced")
            torch.save(model.state_dict(), f'deep_emotion-{args.batch_size}-{lr}-{args.epochs}.pt')

        scheduler.step(avg_vloss)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-e', '--epochs', type=int, default=2)
    parser.add_argument('-b', '--batch_size', type=int, default=64)


    args = parser.parse_args()
    train(args)
