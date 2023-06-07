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

# create models folder if it doesn't exist
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# Hyperparameters
masked_aux = False
max_epochs = 100
num_classes = 8
batch_size = 8
img_size = (256, 256)
learning_rate = 0.0001

def main():
    # Load Data
    train_dataset1 = CityscapesDataset(split='train', img_size=img_size, rotate=True)
    train_dataset2 = CityscapesDataset(split='train', img_size=img_size, rotate=True, foggy=True)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = CityscapesDataset(split='test', img_size=img_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = UNet(num_class=num_classes, out_sz=img_size, masked_aux=masked_aux).to(device)
    # print number of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    # model is pytorch resnet
    # model = torchvision.models.resnet18(pretrained=False)
    # model.fc = torch.nn.Linear(512, 4)
    # model = model.to(device)

    # load model if it exists
    # if os.path.exists('checkpoints/model-7000.ckpt'):
    #     model.load_state_dict(torch.load('checkpoints/model-7000.ckpt'))
    #     print('Model loaded from checkpoints/model.ckpt')

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    criterion_mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create tensorboard writer
    writer = SummaryWriter()

    # Train the model
    log_steps = 50
    class_loss_total = 0
    rot_loss_total = 0
    total_step = len(train_loader)
    step = 0
    for epoch in range(max_epochs):
        for i, (images, labels, rot_label) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            rot_label = rot_label.to(device)
            # import IPython ; IPython.embed() ; exit(1)

            if masked_aux:
                images_label = images
                # mask = train_dataset.sample_mask().to(device)
                # images = images * (1 - mask)

            # Forward pass
            outputs = model(images)
            # rot_pred = outputs
            out_pred = outputs[0]
            rot_pred = outputs[1]

            class_loss = criterion(out_pred, labels)
            if masked_aux:
                rot_loss = criterion_mse(rot_pred, images_label)
            else:
                rot_loss = criterion(rot_pred, rot_label)

            # loss = class_loss + rot_loss
            loss = class_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the loss
            class_loss_total += class_loss.item()
            # class_loss_total = 0
            rot_loss_total += rot_loss.item()

            step += 1

            if step % log_steps == 0:
                total_loss = class_loss_total + rot_loss_total
                writer.add_scalar('class_loss', class_loss_total / log_steps, epoch * total_step + i)
                writer.add_scalar('rot_loss', rot_loss_total / log_steps, epoch * total_step + i)
                writer.add_scalar('total_loss', total_loss / log_steps, epoch * total_step + i)
                # print loss including rotation loss and class loss
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Class Loss: {:.4f}, Rot Loss: {:.4f}'
                        .format(epoch + 1, max_epochs, (step % total_step), total_step, total_loss / log_steps, class_loss_total / log_steps, rot_loss_total / log_steps))
                # print(torch.softmax(rot_pred, dim=1))
                # print(rot_label)

                class_loss_total = 0
                rot_loss_total = 0

            if step % 2500 == 0:
                # evaluate train accuracy
                total_accuracy = 0
                train_dataset.rotate = False
                with torch.no_grad():
                    for i, (images, labels, rot_label) in enumerate(tqdm(train_loader)):
                        images = images.to(device)
                        labels = labels.to(device)
                        rot_label = rot_label.to(device)
                        outputs = model(images)
                        out_pred = outputs[0]
                        rot_pred = outputs[1]
                        class_labels = torch.Tensor(np.argmax(labels.cpu().numpy(), 1)).to(device)
                        _, predicted = torch.max(out_pred.data, 1)
                        total_accuracy += (predicted == class_labels).sum().item() / (img_size[0] * img_size[1])

                train_accuracy = total_accuracy / len(train_dataset)
                print('Train Accuracy: {:.4f}'.format(train_accuracy))
                writer.add_scalar('train_accuracy', train_accuracy, step)
                torch.save(model.state_dict(), os.path.join('checkpoints', 'model-{}.ckpt'.format(step)))

                if train_accuracy > 0.9:
                    print("Final step: {}".format(step))
                    print("Final train accuracy: {}".format(train_accuracy))
                    exit()
                train_dataset.rotate = True

    print("Final step: {}".format(step))
    print("Final train accuracy: {}".format(train_accuracy))
    torch.save(model.state_dict(), os.path.join('checkpoints', 'model-{}.ckpt'.format(step)))

if __name__ == "__main__":
    main()

# model-27500-segonly-rotaug.ckpt : 90.06% train accuracy - epoch 75
# model-32500-joint-auto : Train Accuracy: 0.9026 - epoch 88
# model-37200-joint-auto-mask : Train Accuracy: 0.8953 - epoch 100


# model-35000-segonly-clearfog.ckpt : 89.12% train accuracy - epoch 50 (100)
# model-32500-segonly-rotaug.ckpt : 90.26% train accuracy - epoch 88
# model-32500-joint-auto.ckpt : 90.06% train accuracy - epoch 88