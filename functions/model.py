import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Add positional encoding to input features for transformers"""
    def __init__(self, d_model, dropout=0.1, max_len=101):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)
        position=torch.arange(max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe=torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2]=torch.sin(position * div_term)
        pe[:, 0, 1::2]=torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x=x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimeSeriesTransformerEncoder(nn.Module):
    """Transformer encoder for time series data"""
    def __init__(self, input_dim, hidden_dim, nheads, num_layers, dropout=0.5):
        super().__init__()
        self.encoder=nn.Linear(4, input_dim)
        self.pos_encoder=PositionalEncoding(input_dim, dropout)
        encoder_layers=nn.TransformerEncoderLayer(input_dim, nheads, hidden_dim, dropout)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        src=self.encoder(src)
        src=self.pos_encoder(src)
        output=self.transformer_encoder(src)
        return output

class UNetDecoder(nn.Module):
    """U-Net style decoder for upsampling and processing features"""
    def __init__(self, in_channels):
        super().__init__()
        self.upconv1=nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
        self.conv1=nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=6)
        self.upconv2=nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=2)
        self.conv2=nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=6)
        self.upconv3=nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=3, stride=2)
        self.conv3=nn.Conv2d(in_channels // 8, in_channels // 8, kernel_size=6)
        self.upconv4=nn.ConvTranspose2d(in_channels // 8, in_channels // 16, kernel_size=3, stride=2)
        self.conv4=nn.Conv2d(in_channels // 16, in_channels // 16, kernel_size=6)
        self.final_conv=nn.Conv2d(in_channels // 16, 1, kernel_size=1)

    def forward(self, x):
        x=self.upconv1(x)
        x=self.conv1(x)
        x=self.upconv2(x)
        x=self.conv2(x)
        x=self.upconv3(x)
        x=self.conv3(x)
        x=self.upconv4(x)
        x=self.conv4(x)
        x=self.final_conv(x)
        return x

class TimeSeriesToImageModelCont(nn.Module):
    """Model to process time series data and decode into an image"""
    def __init__(self, input_dim=128, hidden_dim=256, nheads=2, num_layers=2):
        super().__init__()
        self.encoder=TimeSeriesTransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim, nheads=nheads, num_layers=num_layers)
        self.decoder=UNetDecoder(input_dim)

    def forward(self, src):
        src=self.encoder(src)
        src=src.view(src.size(0), -1, 10, 10)
        output=self.decoder(src)
        return output
