import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from PIL import Image
import io
# Defining Class for Single Layer.
class Layer(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1):
        super(Layer,self).__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding)
        self.bn=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU()
        nn.init.xavier_uniform_(self.conv.weight)
    def forward(self,Input):
        output=self.conv(Input)
        output=self.bn(output)
        output=self.relu(output)
        return output

# Define complete model
class CelebModel(nn.Module):
    def __init__(self,num_classes=40):
        super(CelebModel,self).__init__()

        self.unit1=Layer(in_ch=3,out_ch=32)
        self.unit2=Layer(in_ch=32,out_ch=32)
        self.pool1=nn.MaxPool2d(kernel_size=2)

        self.unit3=Layer(in_ch=32,out_ch=64)
        self.unit4=Layer(in_ch=64,out_ch=64)
        self.pool2=nn.MaxPool2d(kernel_size=2)

        self.unit5=Layer(in_ch=64,out_ch=128)
        self.unit6=Layer(in_ch=128,out_ch=128)
        self.unit7=Layer(in_ch=128,out_ch=128)
        self.pool3=nn.MaxPool2d(kernel_size=2)

        self.unit8=Layer(in_ch=128,out_ch=256,kernel_size=5,padding=0)
        self.unit9=Layer(in_ch=256,out_ch=256,kernel_size=5,padding=0)
        self.unit10=Layer(in_ch=256,out_ch=256,kernel_size=5,padding=0)
        self.pool4=nn.MaxPool2d(kernel_size=2)

        self.drop2=nn.Dropout(0.5)

        self.unit11=Layer(in_ch=256,out_ch=512,kernel_size=3,padding=0)
        self.unit12=Layer(in_ch=512,out_ch=512,kernel_size=3,padding=0)
        self.unit13=Layer(in_ch=512,out_ch=512,kernel_size=3,padding=0)

        self.pool5=nn.AvgPool2d(kernel_size=2)

        self.drop3=nn.Dropout(0.5)

        self.model=nn.Sequential(self.unit1,self.unit2,self.pool1,self.unit3,
                                 self.unit4,self.pool2,self.unit5,self.unit6,
                                 self.unit7,self.pool3,self.unit8,self.unit9,
                                 self.unit10,self.pool4,self.drop2,self.unit11,
                                 self.unit12,self.unit13,self.pool5,self.drop3)

        self.fc=nn.Linear(in_features=512,out_features=num_classes)

    def forward(self,Input):

        output=self.model(Input)
        output=output.view(-1,512)
        output=self.fc(output)

        return output

# creating model object
model=CelebModel()

# load weights
model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu')))

# set dropout and batch normalization layers to evaluation mode
model.eval()


def transform_image(image_bytes):
    transform=transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5063, 0.4258, 0.3832],std=[0.2644, 0.2436, 0.2397])])
    image=Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image)


def Predict(image):
    pred=model(image.unsqueeze(0))
    labels=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    attr=list(torch.round(torch.sigmoid(pred)).cpu().detach().numpy().squeeze(0))
    prd=list(torch.sigmoid(pred).cpu().detach().numpy().squeeze(0))
    new_labels=[label for label,a in list(zip(labels,attr)) if a==1]
    pred_list=[p for p,a in list(zip(prd,attr)) if a==1]
    return [{'name':label ,'percent':round(p,2)} for label,p in zip(new_labels,pred_list)]
