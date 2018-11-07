import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
from skimage import io, color
import numpy as np





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(1,8,3,2,1),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(8,16,3,2,1),
            nn.ReLU(),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(16,32,3,2,1),
            nn.ReLU(),
        )   

        self.encoder4 = nn.Sequential(
            nn.Conv2d(32,64,3,2,1),
            nn.ReLU(),
        )   

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32,16,4,2,1),
            nn.ReLU(),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(16,8,4,2,1),
            nn.ReLU(),
        )

        self.decoder4= nn.Sequential(
            nn.ConvTranspose2d(8,1,4,2,1),
            nn.Sigmoid(),
        )
        


    def forward(self, x):
        encoded1= self.encoder1(x)
        encoded2= self.encoder2(encoded1)
        encoded3= self.encoder3(encoded2)
        encoded4= self.encoder4(encoded3)
        
        decoded1= self.decoder1(encoded4)
        decoded1= decoded1+encoded3

        decoded2= self.decoder2(decoded1)
        decoded2= decoded2+encoded2

        decoded3= self.decoder3(decoded2)
        decoded3= decoded3+encoded1


        decoded4 = self.decoder4(decoded3)
        # out = decoded+x
        out= decoded4
        return out

### Create an instance of the Net class
net = Net()

## Loading the training and test sets

# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
transform = transforms.ToTensor()

trainset = torchvision.datasets.STL10(root='./data', split='unlabeled', download=False, transform=transform)
testset = torchvision.datasets.STL10(root='./data', split='test', download=False, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

### Define the loss and create your optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

print(len(trainset))


### Testing the network on  test images and computing the loss
def testing():
    loss=0
    with torch.no_grad():
        for i,data in enumerate(testloader,0):
            input, dummy = data
            target = input


            for j in range(len(input)):
                rgb=input[j]
                rgb=np.array(rgb.data)
                rgb=np.transpose(rgb)
                lab= color.rgb2lab(rgb)
                lab=np.transpose(lab)
                lab=torch.from_numpy(lab)
                lab[:,:,0:1]=lab[:,:,0:1]/100            
                # lab=lab.transpose(0,2)
                input[j].data=lab[:,:,:]

            # input[:,0:1,:,:]= input[:,0:1,:,:]/100

            # target=input[:,0:1,:,:] 

            # preparing the data by downsampling and then upsampling
            image1=torch.nn.functional.interpolate(input,size=48,mode='bilinear')
            input=torch.nn.functional.interpolate(image1,size=96,mode='bilinear')

            #preparing output
            output_lab= input[:,0:3,:,:]
            output_l= net(input[:,0:1,:,:]) 
            output_l=output_l*100
            output_lab[:,0:1,:,:]=output_l[:,0:1,:,:]
            output=output_lab
            # get some random training images
            # images=input
            # images=images.numpy()

            for k in range(len(output)):
                output1=output[k]
                output1=np.array(output1.data)
                output1=np.transpose(output1)
                images1=color.lab2rgb(output1.astype(np.float64))
                images1=np.transpose(images1)
                output[k]=torch.from_numpy(images1)  

            # output=output*255
        
            loss = loss_func(output, target)      # mean square error

            if i==19:
                break

        print("Loss in Test Results....")
        print(loss)   


### Main training loop
for epoch in range(5):
    
    for i, data in enumerate(trainloader, 0):
        
		## Getting the input and the target from the training set
        image, dummy = data


        for j in range(len(image)):
            rgb=image[j]
            rgb=np.array(rgb.data)
            rgb=np.transpose(rgb)
            lab= color.rgb2lab(rgb)
            lab=np.transpose(lab)
            lab=torch.from_numpy(lab)
            lab[:,:,0:1]=lab[:,:,0:1]/100
            # lab=lab.transpose(0,2)
            image[j].data=lab[:,:,:]

        # print(lab)
        # print(image.data)
        #set ground truth before downsampling
        target = image[:,0:1,:,:]    

        # preparing the data by downsampling and then upsampling
        image1=torch.nn.functional.interpolate(image,size=48,mode='bilinear')
        image2=torch.nn.functional.interpolate(image1,size=96,mode='bilinear')
    

        input=image2[:,0:1,:,:]
        # target=target
        pred= net(input.data)
        # print(i)
        loss = loss_func(pred, target)      # mean square error
        if i%500==0 :
              testing()


        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        




### Displaying or saving the results as well as the ground truth images for the first five images in the test set

#show image
def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

for i, data in enumerate(testloader, 0):
        input, dummy = data
        target = input
        
        
        for j in range(len(input)):
            rgb=input[j]
            rgb=np.array(rgb.data)
            rgb=np.transpose(rgb)
            lab= color.rgb2lab(rgb)
            lab=np.transpose(lab)
            lab=torch.from_numpy(lab)
            lab[:,:,0:1]=lab[:,:,0:1]/100
            # lab=lab.transpose(0,2)
            input[j].data=lab[:,:,:]
        
        # input[:,0:1,:,:]= input[:,0:1,:,:]/100

        # preparing the data by downsampling and then upsampling
        image1=torch.nn.functional.interpolate(input,size=48,mode='bilinear')
        input=torch.nn.functional.interpolate(image1,size=96,mode='bilinear')


        #prepare output images

        output_lab= input[:,0:3,:,:]
        output_l= net(input[:,0:1,:,:]) 
        output_l=output_l*100
        output_lab[:,0:1,:,:]=output_l[:,0:1,:,:]
        output=output_lab
       # get some random training images
        # images=input
        # images=images.numpy()
        for k in range(len(output)):
            output1=output[k]
            output1=np.array(output1.data)
            output1=np.transpose(output1)
            images1=color.lab2rgb(output1.astype(np.float64))
            images1=np.transpose(images1)
            output[k]=torch.from_numpy(images1[:,:,:])
        
        images=output

        # print("input...")
        # print(input)
        # print("output...")
        # print(output)
       # show images
        imshow(torchvision.utils.make_grid(images))
        if(i==5):
            break
        # print labels

        print(' '.join())
