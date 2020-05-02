import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from net import Net
from dataset import get_caption_dataset


def main():
    batch_size = 8
    image_size = 224
    hidden_dim = 128 #512
    num_layers = 3 #1

    data_root = "/home/abinaya/Documents/Show_and_tell_pytorch/codes/show_and_tell/data"
    ckpt_path = "/home/abinaya/Documents/Show_and_tell_pytorch/codes/show_and_tell/checkpoint_resnet18_3DRNN/caption_100.pth"

    # to initialize vocab
    _, _, TEXT = get_caption_dataset(
        train=True,
        data_root=data_root,
        batch_size=batch_size, image_size=image_size,
        text_field=True)

    loader, dataset = get_caption_dataset(
        train=False,
        data_root=data_root,
        batch_size=batch_size, image_size=image_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(TEXT, "resnet-18", hidden_dim, num_layers).to(device)

    # load pre-trained model
    state_dict = torch.load(ckpt_path)
    net.load_state_dict(state_dict)

    net.eval()
    cap_repr, pred_repr = list(), list()
    with torch.no_grad():
        for inputs in loader:
            image   = inputs[0].to(device)
            caption = inputs[1]
            #print(image)
            preds = net.sample(image)
            print(preds)
            for pred, cap in zip(preds, caption):
                cap_sent, cap_words = dataset.indices_to_string(cap, True)
                pred_sent, pred_words = dataset.indices_to_string(pred, True)
                
                cap_repr.append(cap_sent)
                pred_repr.append(pred_sent)
                
            break # run only first batch

if __name__ == '__main__':
    main()