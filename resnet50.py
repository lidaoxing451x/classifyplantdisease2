from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle

plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'example'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cpu")
#loaddata

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    train_data={'train':[],'val':[]}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            confusion_matirx={'tp':0.0,'fp':0.0,'fn':0.0,'tn':0.0}
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                confusion_change=count_confusion_matrix(preds,labels)
                for key in confusion_change.keys():
                    confusion_matirx[key]+=confusion_change[key]
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_recall=confusion_matirx['tp']/(confusion_matirx['tp']+confusion_matirx['fn'])
            epoch_precision=confusion_matirx['tp']/(confusion_matirx['tp']+confusion_matirx['fp'])
            epoch_f1=(2*epoch_precision*epoch_recall/(epoch_precision+epoch_recall))
            print('{} Loss: {:.4f} Acc: {:.4f} Recall:{:.4f} Precision:{:.4f} F1:{:.4f} confusion_matrix:{}'.format(
                phase, epoch_loss, epoch_acc,epoch_recall,epoch_precision,epoch_f1,confusion_matirx))
            train_data[phase].append({'Loss':epoch_loss,'Acc':epoch_acc,'Recall':epoch_recall,'Precision':epoch_precision,'F1':epoch_f1,'confusion_matrix':confusion_matirx})
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    with open('train_data.pickle','wb') as f:
        pickle.dump(train_data,f)
    return model
#fineturing
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# count tp,fp,fn,tn using a batch data's preds and labels(tensor form)
# if you want to knowmore about this function please search confusion matirx
def count_confusion_matrix(preds,labels):
    tp=0
    fp=0
    fn=0
    tn=0
    preds=preds.numpy()
    labels=labels.numpy()
    for idx,target in enumerate(labels):
        if target==0:
            if preds[idx]==target:
                tp=tp+1
            else:fn=fn+1
        else:
            if preds[idx]==target:
                tn=tn+1
            else:
                fp=fp+1
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn}
#begin
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=1)

