import torch
import torch.nn as nn
import numpy as np
from model.convolution_lstm import ConvLSTM

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

def attention(ConvLstm_out):
    attention_w = []
    for k in range(5):
            alpha = torch.mul(ConvLstm_out[k], ConvLstm_out[-1])/5
            m = nn.Softmax(dim=1)
            a = m(alpha).detach().numpy()
            attention_w.append(a)
    attention_w = torch.from_numpy(np.array(attention_w))
    convLstmOut = torch.matmul(attention_w, ConvLstm_out).sum(0)
    return convLstmOut

class CnnEncoder(nn.Module):
    def __init__(self):
        super(CnnEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, (1, 1), 1),
            nn.SELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, (2, 2), 1),
            nn.SELU()
        )    
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, (2, 2), 1),
            nn.SELU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2, (2, 2), 1),
            nn.SELU()
        )
    def forward(self, X):
        conv1_out = self.conv1(X)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        return conv1_out, conv2_out, conv3_out, conv4_out


class Conv_LSTM(nn.Module):
    def __init__(self):
        super(Conv_LSTM, self).__init__()
        self.conv1_lstm = ConvLSTM(input_channels=32, hidden_channels=[32, 32, 32], 
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv2_lstm = ConvLSTM(input_channels=64, hidden_channels=[64, 64, 64], 
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv3_lstm = ConvLSTM(input_channels=128, hidden_channels=[128, 128, 128], 
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv4_lstm = ConvLSTM(input_channels=256, hidden_channels=[256, 256, 256], 
                                   kernel_size=3, step=5, effective_step=[4])


    def forward(self, conv1_out, conv2_out, 
                conv3_out, conv4_out):
        conv1_lstm_out = self.conv1_lstm(conv1_out)
        conv1_lstm_out = attention(conv1_lstm_out)
        conv2_lstm_out = self.conv2_lstm(conv2_out)
        conv2_lstm_out = attention(conv2_lstm_out)
        conv3_lstm_out = self.conv3_lstm(conv3_out)
        conv3_lstm_out = attention(conv3_lstm_out)
        conv4_lstm_out = self.conv4_lstm(conv4_out)
        conv4_lstm_out = attention(conv4_lstm_out)
        return conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out

class CnnDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CnnDecoder, self).__init__()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=128, kernel_size=2, stride=[2, 2], padding=0),
            nn.SELU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=[2, 2], padding=1),
            nn.SELU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=[2, 2]),
            nn.SELU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=[1, 1], padding=1),
            nn.SELU()
        )
    
    def forward(self, conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
        deconv4 = self.deconv4(conv4_lstm_out)
        deconv4_concat = torch.cat((deconv4, conv3_lstm_out), dim = 1)
        deconv3 = self.deconv3(deconv4_concat)
        deconv3_concat = torch.cat((deconv3, conv2_lstm_out), dim = 1)
        deconv2 = self.deconv2(deconv3_concat)
        deconv2_concat = torch.cat((deconv2, conv1_lstm_out), dim = 1)
        deconv1 = self.deconv1(deconv2_concat)
        return deconv1

class Generator(nn.Module):
    def __init__(self, in_channels_encoder, in_channels_decoder):
        super(Generator,self).__init__()
        self.cnn_encoder = CnnEncoder(in_channels_encoder)
        self.conv_lstm = Conv_LSTM()
        self.cnn_decoder = CnnDecoder(in_channels_decoder)
        self.second_encoder = CnnEncoder(in_channels_encoder)

    
    def forward(self, x):
        conv1_out, conv2_out, conv3_out, conv4_out = self.cnn_encoder(x)
        conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out = self.conv_lstm(
                                conv1_out, conv2_out, conv3_out, conv4_out)
        latent_z = {"z_out1":conv1_lstm_out, "z_out2": conv2_lstm_out, 
                    "z_out3": conv3_lstm_out, "z_out4":conv4_lstm_out}
        gen_x = self.cnn_decoder(conv1_lstm_out, conv2_lstm_out, 
                                conv3_lstm_out, conv4_lstm_out)
        conv1_out_hat, conv2_out_hat, conv3_out_hat, conv4_out_hat = self.cnn_encoder(gen_x)
        conv1_lstm_out_hat, conv2_lstm_out_hat, conv3_lstm_out_hat, conv4_lstm_out_hat = self.conv_lstm(
                                conv1_out_hat, conv2_out_hat, conv3_out_hat, conv4_out_hat)
        latent_z_hat = {"z_hat_out1":conv1_lstm_out_hat, "z_hat_out2": conv2_lstm_out_hat, 
                    "z_hat_out3": conv3_lstm_out_hat, "z_hat_out4":conv4_lstm_out_hat}
        return latent_z, gen_x, latent_z_hat


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        validity = self.model(x)
        return validity



