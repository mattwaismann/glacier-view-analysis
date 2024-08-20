# Convolutional Model Architecture

import torch
import torch.nn as nn
import torchvision.models as models

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_class=2, freeze_encoder=True):

        # Number of Classes
        self.n_class = n_class
        # Should pre-trained encoder weights be trained or not
        self.freeze_encoder = freeze_encoder
        super(UNet, self).__init__()

        # Loading resnet-50 pre-trained model
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # Freeze encoder weights if set to True
        if self.freeze_encoder:
            for i, param in enumerate(resnet.parameters()):
                if i == 0:
                    pass
                else:
                    param.requires_grad = False
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Replacing encoder's first layer to allow 10 channels input intead of just 3
        self.resnet[0] = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,bias=False) #CHANGE HERE

        self.relu = nn.ReLU(inplace=True)


        # Defining decoder layer layers
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.c1 =  conv_block(2048,1024)#.to(device)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.c2 = conv_block(1024, 512)#.to(device)


        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.c3 = conv_block(512, 256)#.to(device)

        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.c4 = conv_block(128, 64)#.to(device)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.c5 = conv_block(34, 16)#.to(device) #CHANGE HERE
        self.deconv6 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.deconv7 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.deconv8 = nn.ConvTranspose2d(16, 4, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(4)
        self.c8 = nn.Conv2d(4, 4, 6, stride=8, padding=0)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Conv2d(16, self.n_class, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, images):
        x0 = self.resnet[0](images)
        x1 = self.resnet[1](x0)
        x2 = self.resnet[2](x1)
        x3 = self.resnet[3](x2)
        x4 = self.resnet[4](x3)
        x5 = self.resnet[5](x4)
        x6 = self.resnet[6](x5)
        out = self.resnet[7](x6)

        y1 = self.bn1(self.relu(self.deconv1(out)))
        d1 = y1
        y1 = torch.cat([y1, x6], dim=1)
        y1 = self.c1(y1)


        y2 = self.bn2(self.relu(self.deconv2(y1)))
        d2 = y2
        y2 = torch.cat([y2, x5], dim=1)
        y2= self.c2(y2)

        y3 = self.bn3(self.relu(self.deconv3(y2)))
        d3 = y3
        y3 = torch.cat([y3, x4], dim=1)
        y3= self.c3(y3)


        y4 = self.bn4(self.relu(self.deconv4(y3)))
        d4 = y4
        y4 = torch.cat([y4, x2], dim=1)
        y4= self.c4(y4)

        y5 = self.bn5(self.relu(self.deconv5(y4)))
        d5 = y5
        y5 = torch.cat([y5, images], dim=1)
        y5 = self.c5(y5)

        score = self.classifier(y5)

        return score
        
# Convolutional Model Architecture
class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
    
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # print(g1.shape, x1.shape)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class AUNet(nn.Module):
    def __init__(self, n_class, freeze_encoder=True):
        # Number of Classes
        self.n_class = n_class
        # Should pre-trained encoder weights be trained or not
        self.freeze_encoder = freeze_encoder
        super(AUNet, self).__init__()

        # Loading resnet-50 pre-trained model
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # Freeze encoder weights if set to True
        if self.freeze_encoder:
            for i, param in enumerate(resnet.parameters()):
                if i == 0:
                    pass
                else:
                    param.requires_grad = False
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        self.imgs = nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1,bias=False)
        # Replacing encoder's first layer to allow 10 channels input intead of just 3
        self.resnet[0] = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.Sigmoid()


        # Defining decoder layer layers
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        # self.c1 =  conv_block(2048,1024)
        self.c1 = RRCNN_block(2048,1024)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        # self.c2 = conv_block(1024, 512)
        self.c2 = RRCNN_block(1024, 512)


        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # self.c3 = conv_block(512, 256)
        self.c3 = RRCNN_block(512, 256)

        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # self.c4 = conv_block(128, 64)
        self.c4 = RRCNN_block(128, 64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        # self.c5 = conv_block(64, 16)
        self.c5 = RRCNN_block(64, 16)

        # self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Conv2d(16, self.n_class, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.attention1 = AttentionGate(F_g=1024, F_l=1024, F_int=1024)
        self.attention2 = AttentionGate(F_g=512, F_l=512, F_int=512)
        self.attention3 = AttentionGate(F_g=256, F_l=256, F_int=256)
        self.attention4 = AttentionGate(F_g=64, F_l=64, F_int=64)
        self.attention5 = AttentionGate(F_g=32, F_l=32, F_int=32)
        


    def forward(self, images):
        x = self.imgs(images)
        x0 = self.resnet[0](x)
        x1 = self.resnet[1](x0)
        x2 = self.resnet[2](x1)
        x3 = self.resnet[3](x2)
        x4 = self.resnet[4](x3)
        x5 = self.resnet[5](x4)
        x6 = self.resnet[6](x5)
        out = self.resnet[7](x6)
        # print(images.shape)
        # print(x.shape)
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        # print(x6.shape)

        y1 = self.bn1(self.relu(self.deconv1(out)))
        x6 = self.attention1(g=y1, x=x6)
        y1 = torch.cat([y1, x6], dim=1)
        y1 = self.c1(y1)


        y2 = self.bn2(self.relu(self.deconv2(y1)))
        x5 = self.attention2(g=y2, x=x5)
        y2 = torch.cat([y2, x5], dim=1)
        y2= self.c2(y2)

        y3 = self.bn3(self.relu(self.deconv3(y2)))
        x4 = self.attention3(g=y3, x=x4)
        y3 = torch.cat([y3, x4], dim=1)
        y3= self.c3(y3)


        y4 = self.bn4(self.relu(self.deconv4(y3)))
        x2 = self.attention4(g=y4, x=x2)
        y4 = torch.cat([y4, x2], dim=1)
        y4= self.c4(y4)

        y5 = self.bn5(self.relu(self.deconv5(y4)))
        x = self.attention5(g=y5, x=x)
        y5 = torch.cat([y5, x], dim=1)
        y5 = self.c5(y5)

        score = self.classifier(y5)

        return score


class AUNet_NR(nn.Module):
    def __init__(self, n_class, freeze_encoder=True):
        # Number of Classes
        self.n_class = n_class
        # Should pre-trained encoder weights be trained or not
        self.freeze_encoder = freeze_encoder
        super(AUNet_NR, self).__init__()

        # Loading resnet-50 pre-trained model
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        # Freeze encoder weights if set to True
        if self.freeze_encoder:
            for i, param in enumerate(resnet.parameters()):
                if i == 0:
                    pass
                else:
                    param.requires_grad = False
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        self.imgs = nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1,bias=False)
        # Replacing encoder's first layer to allow 10 channels input intead of just 3
        self.resnet[0] = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.Sigmoid()


        # Defining decoder layer layers
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.c1 =  conv_block(2048,1024)
        # self.c1 = RRCNN_block(2048,1024)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.c2 = conv_block(1024, 512)
        # self.c2 = RRCNN_block(1024, 512)


        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.c3 = conv_block(512, 256)
        # self.c3 = RRCNN_block(512, 256)

        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.c4 = conv_block(128, 64)
        # self.c4 = RRCNN_block(128, 64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.c5 = conv_block(64, 16)
        # self.c5 = RRCNN_block(64, 16)

        # self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Conv2d(16, self.n_class, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.attention1 = AttentionGate(F_g=1024, F_l=1024, F_int=1024)
        self.attention2 = AttentionGate(F_g=512, F_l=512, F_int=512)
        self.attention3 = AttentionGate(F_g=256, F_l=256, F_int=256)
        self.attention4 = AttentionGate(F_g=64, F_l=64, F_int=64)
        self.attention5 = AttentionGate(F_g=32, F_l=32, F_int=32)


    def forward(self, images):
        x = self.imgs(images)
        x0 = self.resnet[0](x)
        x1 = self.resnet[1](x0)
        x2 = self.resnet[2](x1)
        x3 = self.resnet[3](x2)
        x4 = self.resnet[4](x3)
        x5 = self.resnet[5](x4)
        x6 = self.resnet[6](x5)
        out = self.resnet[7](x6)

        y1 = self.bn1(self.relu(self.deconv1(out)))
        x6 = self.attention1(g=y1, x=x6)
        y1 = torch.cat([y1, x6], dim=1)
        y1 = self.c1(y1)


        y2 = self.bn2(self.relu(self.deconv2(y1)))
        x5 = self.attention2(g=y2, x=x5)
        y2 = torch.cat([y2, x5], dim=1)
        y2= self.c2(y2)

        y3 = self.bn3(self.relu(self.deconv3(y2)))
        x4 = self.attention3(g=y3, x=x4)
        y3 = torch.cat([y3, x4], dim=1)
        y3= self.c3(y3)


        y4 = self.bn4(self.relu(self.deconv4(y3)))
        x2 = self.attention4(g=y4, x=x2)
        y4 = torch.cat([y4, x2], dim=1)
        y4= self.c4(y4)

        y5 = self.bn5(self.relu(self.deconv5(y4)))
        x = self.attention5(g=y5, x=x)
        y5 = torch.cat([y5, x], dim=1)
        y5 = self.c5(y5)

        score = self.classifier(y5)

        return score