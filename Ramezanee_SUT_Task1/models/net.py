import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class RepConv2d(nn.Module):
    def __init__(self, input_channel, output_channel, stride=(1, 1), groups=1):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.stride = stride

        # Initial convolutions
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3),
                               stride=stride, padding=(1, 1), bias=False,groups=groups)
        self.conv2 = nn.Conv2d(input_channel, output_channel, kernel_size=(1, 3),
                               stride=stride, padding=(0, 1), bias=False,groups=groups)
        self.conv3 = nn.Conv2d(input_channel, output_channel, kernel_size=(3, 1),
                               stride=stride, padding=(1, 0), bias=False,groups=groups)
        self.conv4 = nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1),
                               stride=stride, padding=(0, 0), bias=False,groups=groups)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4

    def merge_convs(self):
        conv2ds_list = [self.conv1, self.conv2, self.conv3, self.conv4]
        main_shape = conv2ds_list[0].weight.data.shape
        device = conv2ds_list[0].weight.device  # Get device from one of the convs

        # Initialize zero tensors on the correct device
        conv1 = conv2ds_list[0].weight.data
        conv2 = torch.zeros(main_shape, device=device)
        conv3 = torch.zeros(main_shape, device=device)
        conv4 = torch.zeros(main_shape, device=device)

        # Fill the corresponding parts
        conv2[:, :, 1, :] = conv2ds_list[1].weight.data.squeeze(2)
        conv3[:, :, :, 1] = conv2ds_list[2].weight.data.squeeze(3)
        conv4[:, :, 1, 1] = conv2ds_list[3].weight.data.squeeze(3).squeeze(2)

        # Create new Conv2d layer on same device
        conv2d = nn.Conv2d(
            in_channels=self.conv1.in_channels,
            out_channels=self.conv1.out_channels,
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            bias=False,
            groups=self.conv1.groups
        ).to(device)

        with torch.no_grad():
            conv2d.weight.copy_((conv1 + conv2 + conv3 + conv4))

        return conv2d

    def get_reparametrized_layer(self):
        conv2d = self.merge_convs()
        return nn.Sequential(conv2d)


class ResidualNormalization(nn.Module):
    """
    Combined normalization layer:
    λ * x + InstanceNorm(x)
    """
    def __init__(self, num_features):
        super().__init__()
        # Learnable per-channel scaling factor
        self.lambda_param = nn.Parameter(torch.ones(num_features, 1, 1))
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x):
        return self.lambda_param * x + self.instance_norm(x)


class LearnablePooling(nn.Module):
    """
    Attention-based learnable pooling with Global Average Pooling (GAP)
    Output: concat(attention_pooled_features, GAP_features)
    """
    def __init__(self, in_channels, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_channels // 2
        
        # Input normalization
        self.bn_input = ResidualNormalization(in_channels)
        
        # Attention mechanism
        self.attn_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_attn = ResidualNormalization(hidden_dim)
        self.attn_score = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Input normalization
        x_norm = self.bn_input(x)
        
        # Attention weights calculation
        attn = self.activation(self.bn_attn(self.attn_conv(x_norm)))
        scores = self.attn_score(attn)
        
        # Softmax over spatial dimensions
        b, c, h, w = x.size()
        spatial_weights = F.softmax(scores.view(b, c, -1), dim=-1).view(b, c, h, w)
        
        # Attention-weighted pooling
        attn_pooled = (x * spatial_weights).sum(dim=[2, 3])
        
        # Global average pooling
        gap_pooled = self.global_avg_pool(x).squeeze(-1).squeeze(-1)
        
        # Concatenate both pooling results
        return torch.cat([attn_pooled, gap_pooled], dim=1)


class DSFlexiNetBlock(nn.Module):
    """Inverted Residual Block with Expansion and RepConv"""
    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super().__init__()
        self.stride = stride
        self.use_skip = True
        mid_channels = in_channels * expansion_factor
        
        # ---- Input normalization and scaling ----
        self.input_norm = nn.BatchNorm2d(in_channels)
        self.input_scaling = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False) if self.use_skip else None
        
        # ---- Expansion convolution ----
        self.expand_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = False)
        self.expand_norm = nn.BatchNorm2d( out_channels )
        self.expand_activation = nn.LeakyReLU()
        
        # ---- Spatial convolution ----
        self.spatial_conv = RepConv2d( out_channels , out_channels, stride=stride, groups = out_channels)
        self.spatial_norm = nn.BatchNorm2d( out_channels )
        self.spatial_activation = nn.LeakyReLU()
        
        # ---- Projection convolution ----
        # self.project_conv = nn.Conv2d(out_channels , out_channels , kernel_size = 1 , bias = False ,  )
        # self.project_norm = nn.BatchNorm2d( mid_channels )
        # self.dropout = nn.Dropout2d( 0.1 )

    def forward(self, x):
        residual = self.input_scaling(x) if self.use_skip else None
        
        # Input normalization
        out = self.input_norm(x)
        
        # Expansion
        out = self.expand_conv(out)
        out = self.expand_activation(out)
        
        # Spatial processing
        out = self.spatial_norm(out)
        out = self.spatial_conv(out)
        out = self.spatial_activation(out)
        
        # Projection
        # out = self.project_norm(out)
        # out = self.project_conv(out)
        # out = self.dropout(out)
        
        # Residual connection
        return out + residual if self.use_skip else out


class DSFlexiNet(nn.Module):
    """Main Network Architecture with RepConv and Flexible Blocks"""
    def __init__(self, num_classes=10):
        super().__init__()
        # assert len(expansion_factors) == 6, "Requires 6 expansion factors"
        
        # ---- Initial Convolution Layers ----
        self.input_norm = nn.BatchNorm2d(1)
        
        # Stage 1: Downsample
        self.conv1 = RepConv2d(1, 16, stride = ( 2 , 2))
        self.norm1 = nn.BatchNorm2d(16)
        self.activation1 = nn.ReLU()
        # Stage 2: Downsample
        self.conv2 = RepConv2d( 16 , 32 , stride = ( 2 , 2) )
        self.activation2 = nn.ReLU()
        
        # ---- Residual Stages ----
        # Stage 1: Residual blocks
        self.stage1 = nn.Sequential(
            DSFlexiNetBlock(32, 32, stride=(1,1)),
            DSFlexiNetBlock(32, 32, stride=(1,1)),
            DSFlexiNetBlock(32, 32, stride=(1,1))
        )
        self.stage1_norm = ResidualNormalization(32)
        
        # Stage 2: Residual blocks
        self.stage2 = nn.Sequential(
            DSFlexiNetBlock(32, 32, stride=(1,1)),
            DSFlexiNetBlock(32 , 32, stride=(1,1)),
            DSFlexiNetBlock(32 , 32, stride=(1,1)),
        )
        self.stage2_norm = ResidualNormalization(32)
        
        # Stage 3: Final residual block
        self.stage3 = nn.Sequential(
            DSFlexiNetBlock(32 , 64 , stride=(1,1)),
        )
        
        self.stage3_norm = ResidualNormalization(64)
        
        # ---- Classification Head ----
        self.pooling = LearnablePooling(64)
        self.head_norm = nn.BatchNorm1d(64 * 2)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(64 * 2, num_classes)

    def forward(self, x , device = None):
        # Input preprocessing
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        x = self.input_norm(x)
        # Initial convolution stages
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        
        # Residual stages with skip connections
        x = self.stage1(x) + x
        x = self.stage1_norm(x)
        
        x = self.stage2(x) + x
        x = self.stage2_norm(x)
        
        x = self.stage3(x)
        x = self.stage3_norm(x)
        
        # Classification head
        x = self.pooling(x)
        x = self.head_norm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def ReParametrize(module,device):
    """
    Recursively replaces all RepConv2d layers in a module with their reparametrized version.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, RepConv2d):
            # print(f"Reparametrizing {name}")
            new_module = child.get_reparametrized_layer().to(device)
            setattr(module, name, new_module)
        else:
            ReParametrize(child,device)


def get_model(device="cpu"):
    m = DSFlexiNet().to(device)
    ReParametrize(m, device=device)
    return m
