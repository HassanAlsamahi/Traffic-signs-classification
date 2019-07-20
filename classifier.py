import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import torch
import Network_architecture as network
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import dataset
import torchvision.datasets as datasets
import argparse

parser = argparse.ArgumentParser(description = "Import the image here")
parser.add_argument("-i","--image",type=str,required=True, help="Insert the name of the file you want to classify")
parser.add_argument("--model",type=str,required=True, help= "Insert the model")
args = parser.parse_args()
def imshow(img):
    img = img/2 + 0.5
    plt.imshow(np.transpose(img,(1,2,0)))



imsize = 256
loader = transforms.Compose([transforms.Resize((20,20)),
                                transforms.RandomRotation(60),
                                transforms.ToTensor(),
                                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU
imagepath =args.image
image = image_loader(imagepath)
#img = Image.open("traffic-sign.jpg").convert("RGB")
#print(img.shape)
img = mpimg.imread(imagepath)
#img = img.resize((20,20))
#image = datasets.ImageFolder("traffic-sign.jpg",transform=transforms)
#image = imgtensor.unsqueeze_(0)
model = network.Network()
model_loaded = args.model
model.load_state_dict(torch.load(model_loaded))
output = model(image)
_,preds_tensor = torch.max(output,1)
pred = np.squeeze(preds_tensor.cpu().numpy())
image = image.data.cpu().numpy()
#image.show()
plt.imshow(img)

#figure = plt.figure(figsize = (20,20))
print(dataset.classes[pred])
print(pred)
plt.show()
#ax.set_title('{}',format(dataset.classes[pred]))
#plt.show()
