import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from CityscapesDataloader import CityscapesDataset
from UNet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(1612)

# create models folder if it doesn't exist
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# Hyperparameters
masked_aux = False
num_classes = 8
batch_size = 1
img_size = (256, 256)
ttt = False
learning_rate = 0.00001

checkpoint_num = "27500-segonly-rotaug"

foggy = True
for fog_level in range(4):
    if fog_level == 3:
        foggy = False
        fog_level = 0
    # Load Data
    train_dataset = CityscapesDataset(split='train', img_size=img_size, rotate=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CityscapesDataset(split='val', img_size=img_size, foggy=foggy, rotate=False, fog_level=fog_level)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = UNet(num_class=num_classes, out_sz=img_size, masked_aux=masked_aux).to(device)
    # print number of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    if os.path.exists(f'checkpoints/model-{checkpoint_num}.ckpt'):
        model.load_state_dict(torch.load(f'checkpoints/model-{checkpoint_num}.ckpt'))
        print(f'Model loaded from checkpoints/model-{checkpoint_num}.ckpt')
    else:
        exit()

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    criterion_mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

    total_accuracy = 0
    total_loss = 0
    for i, (images, labels, rot_label) in enumerate(tqdm(test_loader)):
        images_orig = images.to(device)
        labels_orig = labels.to(device)
        rot_label_orig = rot_label.to(device)

        # perform test time training
        if ttt:
            for _ in range(1):
                if masked_aux:
                    mask = test_dataset.sample_mask().to(device)
                    images = images_orig
                    # images = images_orig * (1 - mask)
                    outputs = model(images)
                    out_pred = outputs[0]
                    rot_pred = outputs[1]
                    rot_label = images_orig
                    rot_loss = criterion_mse(rot_pred, rot_label)
                    # rot_loss = criterion(out_pred, labels_orig)
                    # print(rot_loss)
                else:
                    # sample random rotation
                    img, label, rot_label = test_dataset.sample_rotation(images_orig, labels_orig)
                    outputs = model(images_orig)
                    out_pred = outputs[0]
                    rot_pred = outputs[1]
                    # get rotation loss
                    rot_label = torch.Tensor([rot_label]).to(device).long()
                    rot_loss = criterion(rot_pred, rot_label)
                # plt.imshow(test_dataset.unnormalize(img[0].cpu()))
                # take step
                optimizer.zero_grad()
                rot_loss.backward()
                optimizer.step()

        # Forward pass
        outputs = model(images_orig)
        out_pred = outputs[0]
        rot_pred = outputs[1]

        class_loss = criterion(out_pred, labels_orig)
        if masked_aux:
            rot_loss = criterion_mse(rot_pred, images_orig)
        else:
            rot_loss = criterion(rot_pred, rot_label_orig)

        loss = class_loss + rot_loss
        
        # calculate accuracy
        outputs = torch.nn.functional.softmax(out_pred, dim=1)
        class_labels = torch.Tensor(np.argmax(labels_orig.cpu().numpy(), 1)).to(device)
        _, predicted = torch.max(outputs.data, 1)
        img_accuracy = (predicted == class_labels).sum().item() / (img_size[0] * img_size[1])
        total_accuracy += img_accuracy

        total_loss += loss.item()

        # uncomment for TTT Standard

        # if os.path.exists(f'checkpoints/model-{checkpoint_num}.ckpt'):
        #     model.load_state_dict(torch.load(f'checkpoints/model-{checkpoint_num}.ckpt'))
        #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if fog_level == 0:
        string = "600m"
    elif fog_level == 1:
        string = "300m"
    elif fog_level == 2:
        string = "150m"
    else:
        string = "clear"
    print(f'Accuracy {string}: {total_accuracy / len(test_loader)}')
