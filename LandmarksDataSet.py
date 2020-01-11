import numpy
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import random

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class LandmarksImagesDataSet(Dataset):
    def __init__(self,imageDirectoryPath,landmarkAnnotationsFilePath):
        self.imageDirectoryPath = imageDirectoryPath
        self.landmarkAnnotationsFilePath = landmarkAnnotationsFilePath
        self.datasetList = self.returnDatasetList()


    def returnDatasetList(self):
        annotationsFile = open(self.landmarkAnnotationsFilePath, "r")
        i = 0
        datasetList = []
        for line in annotationsFile:
            if(i>9999):
                break
            i=i+1
            annots = line.split()
            imageName = annots[0]

            label = self.returnImageLabelName(imageName)
            imagePath = self.imageDirectoryPath+'/'+label[0]+'/'+imageName

            #print(label)
            boundaryBox = [float(annots[1]), float(annots[2]), float(annots[3]), float(annots[4])]

            rr = [float(annots[5]), float(annots[6])]
            rl = [float(annots[7]), float(annots[8])]
            lr = [float(annots[9]), float(annots[10])]
            ll = [float(annots[11]), float(annots[12])]
            mouth_corner_r = [float(annots[13]), float(annots[14])]
            mouth_corner_l = [float(annots[15]), float(annots[16])]
            nose = [float(annots[17]), float(annots[18])]
            landmarks = [rr,rl,lr,ll,mouth_corner_r,mouth_corner_l,nose]

            datasetListEntry = [imagePath, boundaryBox, landmarks]
            datasetList.append(datasetListEntry)

        random.shuffle(datasetList)
        return datasetList

    def returnImageLabelName(self, imageName):
        label = imageName.rsplit('_', 1)
        return label


    def display(self,image,updatedLandmarks):
        plt.figure()
        plt.imshow(image)
        plt.scatter(updatedLandmarks[:, 0], updatedLandmarks[:, 1], marker='.', c='r')
        plt.show()


class ItemsDataset(Dataset):
    def __init__(self, passed_data_list,augment):
        """ Initialization: load the dataset list
        """
        self.passed_data_list = passed_data_list
        self.augment = augment

    def __len__(self):
        return len(self.passed_data_list)

    def __getitem__(self, idx):
        item = self.passed_data_list[idx]


        image_path = item[0]
        boundary_box = item[1]


        image = Image.open(item[0]).crop(tuple(boundary_box)).resize((225, 225))

        if (self.augment == 1):
            image = ImageEnhance.Brightness(image).enhance(3.0)

        if (self.augment == 2):
            boundary_box_new = boundary_box
            boundary_box_new[2] = boundary_box_new[2] - 1
            boundary_box_new[3] = boundary_box_new[3] - 1
            image = Image.open(item[0]).crop(tuple(boundary_box_new)).resize((225, 225))

        image_arr = np.asarray(image, dtype=np.float32)
        image_arr_rescaled = (image_arr/255)*2-1
        height, width = image_arr.shape[0], image_arr.shape[1]
        #height, width = image_arr_rescaled.shape[0], image_arr_rescaled.shape[1]

        # Create image tensor
        img_arr_transposed = np.transpose(image_arr)
        #img_arr_transposed = np.transpose(image_arr_rescaled)
        img_tensor = torch.from_numpy(img_arr_transposed)

        # Reshape to (1, 28, 28), the 1 is the channel size
        #img_tensor = img_tensor.view((1, height, width))




        landmarksForIdx = np.asarray(item[2], dtype=np.float32)
        updatedLandmarks = landmarksForIdx - [boundary_box[0],boundary_box[1]]
        updated_image_width = boundary_box[2] - boundary_box[0]
        updated_image_height = boundary_box[3] - boundary_box[1]

        #updatedLandmarks[:, 0] = updatedLandmarks[:, 0] * (225 / updated_image_width)
        #updatedLandmarks[:, 1] = updatedLandmarks[:, 1] * (225 / updated_image_height)

        updatedLandmarks[:, 0] = updatedLandmarks[:, 0] / updated_image_width
        updatedLandmarks[:, 1] = updatedLandmarks[:, 1] / updated_image_height

        label = updatedLandmarks

        label_tensor = torch.from_numpy(label).view((14)).type(torch.float32)  # Loss measurement requires long type tensor


        return img_tensor, label_tensor






