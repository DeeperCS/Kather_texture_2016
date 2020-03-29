import os
import time
import pandas as pd
from glob import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from torch import optim 
from torch.autograd import Variable
import torch.nn.functional as F

# Split

def stratify_split(dataset_pd, label_head, validation_ratio):
    train_pd_set = []
    valid_pd_set = []
    unique_labels = np.unique(dataset_pd[label_head])
    for label in unique_labels:
        flag_this_label = (dataset_pd[label_head]==label)
        number_samples_this_label = sum(flag_this_label)
        number_samples_validation = int(validation_ratio*number_samples_this_label)
        perm = np.random.permutation(number_samples_this_label)
        train_idx = perm[0:-number_samples_validation]
        valid_idx = perm[-number_samples_validation:]
        train_pd_set.append(dataset_pd.loc[flag_this_label].iloc[train_idx])
        valid_pd_set.append(dataset_pd.loc[flag_this_label].iloc[valid_idx])
    train_pd = pd.concat(train_pd_set)
    valid_pd = pd.concat(valid_pd_set)
    return train_pd, valid_pd

def load_data_from_pd(dataset_pd, x_head, y_head):
    data_x_ = []
    data_y_ = []
    for index,row in valid_pd.iterrows():
        file_path = row[x_head]
        label = row[y_head]
        data_x_.append(Image.open(row[x_head]))
        data_y_.append(int(label))
    return data_x_, data_y_


## Network 

def train(train_loader, model, optimizer, epoch, args):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        # import pdb; pdb.set_trace()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_value = loss.item()
        avg_loss += loss_value
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_value))

def test(test_loader, model, args):
    model.eval()
    test_loss = 0
    correct = 0
    pred_all_prob = []
    pred_all_label = []
    
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        
        # import pdb; pdb.set_trace()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        output_prob = F.softmax(output)
        pred_all_prob.extend(output_prob.data.cpu().numpy()[:,0].reshape(-1).tolist())

        pred_all_label.extend(pred.cpu().numpy().reshape(-1).tolist())
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / float(len(test_loader.dataset)), pred_all_label, pred_all_prob

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
        
class TextureDataset(Dataset):
    def __init__(self, data_x, data_y, transform):
        self.data_x = data_x # .transpose([0,3,1,2]) # N, C, H, W
        self.data_y = data_y
        self.transform = transform
        if self.transform:
            self.data_x = [self.transform(x) for x in data_x]
        
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
 
    def __len__(self):
        return len(self.data_y)

# Setting
class Arg():
    def __init__(self):
        pass
    
args = Arg()
args.workers = 4
args.half = False
args.cuda = True
args.test_batch_size = 32
# From paper
args.batch_size = 4
# From paper: 0.0001
args.lr = 0.00001  
args.weight_decay = 1e-4
args.momentum = 0.9

args.start_epoch = 0
args.epochs = 140
args.log_interval = 40

args.save="./models"

# Organizing data using pandas
base_tile_dir = '../../data/Kather_texture_2016_image_tiles_5000/'

file_list = glob(os.path.join(base_tile_dir, '*', '*.tif'))  # 'base_tile_dir/*/*.tif'
print("Number of files:", len(file_list))

all_path = pd.Series(file_list) # Column 1
dataset = pd.DataFrame({'path':all_path})

dataset['cell_type'] = dataset['path'].apply(lambda x: os.path.basename(os.path.dirname(x))) # Column 2
category_names = sorted(list(set(dataset['cell_type'])))
dataset['cell_type_idx'] = dataset['cell_type'].apply(lambda x: category_names.index(x)) # Column 3


# Split the dataset as train and valid sets via pandas
train_pd, valid_pd = stratify_split(dataset, label_head='cell_type_idx', validation_ratio=0.2)

data_x_train, data_y_train = load_data_from_pd(train_pd, x_head='path', y_head='cell_type_idx')
data_x_val, data_y_val = load_data_from_pd(valid_pd, x_head='path', y_head='cell_type_idx')

# Dataset 
textureDatasetTrain = TextureDataset(data_x_train, data_y_train,transform=transforms.Compose([transforms.Resize((224, 224)),
                                                       transforms.ToTensor()]))
textureDatasetVal = TextureDataset(data_x_val, data_y_val,transform=transforms.Compose([transforms.Resize((224, 224)),
                                                       transforms.ToTensor()]))

# Dataloader
train_loader = DataLoader(textureDatasetTrain, batch_size=args.batch_size, shuffle=True, num_workers=1)

test_loader = DataLoader(textureDatasetVal, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

# Build the model
arch = 'resnet18'
model = models.__dict__[arch]()
# Change the number of classes at output layer
num_classes = 8
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)
if args.cuda:
    model = model.cuda()
    
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# assert(0)
best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    begin = time.time()
    print("Epoch begin")
    for param_group in optimizer.param_groups:
        print("Learning rate:", param_group['lr'])
    train(train_loader, model, optimizer, epoch, args)
    print("time cost:", time.time()-begin)
    prec1, pred_all, pred_prob = test(test_loader, model, args)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save)
print("Training finished!")