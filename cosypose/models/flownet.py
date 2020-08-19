# See Implementation here https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetS.py
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from cosypose.config import LOCAL_DATA_DIR


class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self, n_inputs=6, batchNorm=False):
        super(FlowNetS,self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,  n_inputs,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        # self.deconv5 = deconv(1024,512)
        # self.deconv4 = deconv(1026,256)
        # self.deconv3 = deconv(770,128)
        # self.deconv2 = deconv(386,64)

        # self.predict_flow6 = predict_flow(1024)
        # self.predict_flow5 = predict_flow(1026)
        # self.predict_flow4 = predict_flow(770)
        # self.predict_flow3 = predict_flow(386)
        # self.predict_flow2 = predict_flow(194)

        # self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # flow6       = self.predict_flow6(out_conv6)
        # flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        # out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        # concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        # flow5       = self.predict_flow5(concat5)
        # flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        # out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        # concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        # flow4       = self.predict_flow4(concat4)
        # flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        # out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        # concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        # flow3       = self.predict_flow3(concat3)
        # flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        # out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        # concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        # flow2 = self.predict_flow2(concat2)

        return out_conv6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def flownet_pretrained(n_inputs):
    assert n_inputs == 6
    model = FlowNetS(n_inputs=n_inputs).cuda()
    # Download pretrained checkpoint here https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetS.py
    state_dict = torch.load(LOCAL_DATA_DIR / 'flownets_EPE1.951.pth.tar')
    state_dict = {k: v for k, v in state_dict.items() if ('conv' in k and 'deconv' not in k)}
    model.load_state_dict(state_dict)
    return model
