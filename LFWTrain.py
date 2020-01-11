import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from LandmarksDataSet import LandmarksImagesDataSet
from LandmarksDataSet import ItemsDataset
import os
import math

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x) #change this
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

#Define Network Instance
model = alexnet(pretrained = True) #pretrained model
model.cuda()

k=1
for param in model.parameters():
    param.requires_grad = False
    k=k+1
    if(k==5):
        break
classifier_net = list(model.classifier.children())
classifier_net[1] = nn.Linear(256*6*6,512)
classifier_net[4] = nn.Linear(512,256)
classifier_net[6] = nn.Linear(256,14)

# model = AlexNet() #pretrained model
# model.cuda()

model.classifier = nn.Sequential(*classifier_net)


#Define the loss using MSE Loss
criterion = torch.nn.MSELoss()

#Defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_losses = []
valid_losses = []


#File paths
imagesDirectoryPath = "/home/mlyip/sfuhome/CMPT-742/Lab-Session-2/lfw"
trainingAnnotationsFilePath = "/home/mlyip/sfuhome/CMPT-742/Lab-Session-2/LFW_annotation_train.txt"
testingAnnotationsFilePath = "/home/mlyip/sfuhome/CMPT-742/Lab-Session-2/LFW_annotation_test.txt"
LandmarksImagesDataSet(imagesDirectoryPath, trainingAnnotationsFilePath)
LandmarksImagesDataSet(imagesDirectoryPath, testingAnnotationsFilePath)


#Training Dataset 80%
datasetList = LandmarksImagesDataSet(imagesDirectoryPath, trainingAnnotationsFilePath).datasetList
n_training_items = 0.8 * (len(datasetList))
training_items_list = datasetList[: int(n_training_items)]

#Validation Data 20%
n_valid_items = 0.2 * (len(datasetList))
valid_items_list = datasetList[int(n_training_items): int(n_training_items + n_valid_items)]

#Test Dataset
test_dataset = LandmarksImagesDataSet(imagesDirectoryPath, testingAnnotationsFilePath).datasetList
items_dataset_instance = ItemsDataset(test_dataset,0)
length_test_dataset = len(test_dataset)
testing_items_list = datasetList[: int(len(test_dataset))]


#Create DataLoader for training and validation
augments = [0,1,2]
for augment in augments:
    print('augment',augment)
    train_dataset = ItemsDataset(training_items_list, augment)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
    print('Total number of batches:', len(train_data_loader), 'Total training items:', len(train_dataset))

    valid_dataset = ItemsDataset(valid_items_list, augment)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=6)
    print('Total validation items:', len(valid_dataset))

    test_dataset = ItemsDataset(testing_items_list, 0)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Total testing itmes:', len(test_dataset))

    # Visualizing Dataset
    idx, (image_tensor, label_tensor) = next(enumerate(train_data_loader))
    print('Image Tensor Shape (N,C,H,W) :', image_tensor.shape)
    print('Image Label Shape (N, 7, 2) :', label_tensor.shape)

    N, C, H, W = image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]

    nd_image = (image_tensor.cpu().numpy().reshape(N, C, H, W) + 1) / 2.
    nd_label = label_tensor.cpu().numpy().reshape(N, 7, 2)


    max_epoch = 8
    itr = 0

    for epoch_idx in range(0, max_epoch):
        print("inside epoch")
        # iterate the mini batches
        for train_batch_idx, (train_input, train_label) in enumerate(train_data_loader):

            # switch to train model
            model.train()

            # Reset the update parameters to 0
            optimizer.zero_grad()

            # Forward
            train_input = Variable(train_input.cuda())
            train_out = model.forward(train_input)

            # Compute the loss
            train_label = Variable(train_label.cuda())
            loss = criterion(train_out, train_label)

            # Do the backward to compute the gradient flow
            loss.backward()

            # Update the parameters
            optimizer.step()

            train_losses.append((itr, loss.item()))
            itr += 1

            if train_batch_idx % 100 == 0:
                print('[TRAIN] Epoch: %d Iter: %d Loss: %f' % (epoch_idx, itr, loss.item()))

            # Run the validation every 200 iteration
            if train_batch_idx % 200 == 0:
                # Validation
                model.eval()
                valid_losses_subset = []  # collect the validation losses for avg.
                valid_itr = 0

                for valid_batch_idx, (valid_input, valid_label) in enumerate(valid_data_loader):
                    model.eval()
                    valid_input = Variable(valid_input.cuda())
                    valid_out = model.forward(valid_input)

                    # Forward and compute loss
                    valid_label = Variable(valid_label.cuda())
                    valid_loss = criterion(valid_out, valid_label)
                    valid_losses_subset.append(valid_loss.item())  # this had an index 0 in item, check
                    valid_itr += 1
                    if valid_itr > 5:
                        break
                        # avg. valid loss
                avg_loss = np.mean(np.asarray(valid_losses_subset))
                valid_losses.append((itr, avg_loss))
                print('[VALID] Epoch: %d Iter: %d Loss: %f' % (epoch_idx, itr, avg_loss))


#class RunningEpochs():def training_for_each_augmented_dataset(train_data_loader, valid_data_loader):


train_losses = np.asarray(train_losses)
plt.plot(train_losses[:, 0], train_losses[:, 1])
valid_losses = np.asarray(valid_losses)
plt.plot(valid_losses[:, 0], valid_losses[:, 1])
plt.show()

#Save the trained network
net_state = model.state_dict()
model_file_path = '/home/mlyip/sfuhome/CMPT-742/Lab-Session-2/lfw'
torch.save(net_state, os.path.join(model_file_path, "alexnet.pth"))


#   Testing
test_net = alexnet(pretrained=True)
classifier_net = list(model.classifier.children())
classifier_net[1] = nn.Linear(256*6*6,512)
classifier_net[4] = nn.Linear(512,256)
classifier_net[6] = nn.Linear(256,14)

test_net.classifier = nn.Sequential(*classifier_net)
#Load serialized data
test_net_state = torch.load(os.path.join(model_file_path, "alexnet.pth"))

#load weights/parameters from the serialized data
test_net.load_state_dict(test_net_state)

#Set the network to evaluate model
test_net.eval()

# plotting the accuracy or average percentage of detected key points vs radius

distances_l2_average = []

for test_idx, (image_tensor_test, label_tensor_test) in enumerate(test_data_loader):
    # Forward for prediction
    pred = test_net.forward(image_tensor_test.cuda())

    denormalized_pred = (pred.detach().cpu().numpy() * 255) / 2 + 1

    denormalized_label = (label_tensor_test.cpu().numpy() * 255) / 2 + 1

    denormalized_label_extracted = denormalized_label[0]
    denormalized_pred_extracted = denormalized_pred[0]

    distance = 0
    for x in range(0, 7):
        x = x * 2

        label_landmark = np.array(denormalized_label_extracted[x], denormalized_label_extracted[x + 1])
        predicted_landmark = np.array(denormalized_pred_extracted[x], denormalized_pred_extracted[x + 1])
        x_square = (denormalized_label_extracted[x] - denormalized_pred_extracted[x]) ** 2
        y_square = (denormalized_label_extracted[x + 1] - denormalized_pred_extracted[x + 1]) ** 2
        distance = distance + math.sqrt(x_square + y_square)

    average_l2_distance = distance/7

    distances_l2_average.append(average_l2_distance)

total_predictions = len(distances_l2_average)
detected_ratio_vs_radius = []
detected_ratio_list = []

radius_range = [1,3, 5,7,9, 11, 13,15,17,19,21,23,25]

for r in radius_range:

    correct_predictions = 0
    for l2 in distances_l2_average:
        if(l2<=r):
            correct_predictions = correct_predictions + 1

    detected_ratio = correct_predictions/total_predictions

    detected_ratio_arr = [r,detected_ratio]
    detected_ratio_list.append(detected_ratio)
    detected_ratio_vs_radius.append(detected_ratio_arr)

x = radius_range
y = detected_ratio_list

plt.plot(x,y)
# naming the x axis
plt.xlabel('radius')
# naming the y axis
plt.ylabel('Detected ratio')
plt.show()






