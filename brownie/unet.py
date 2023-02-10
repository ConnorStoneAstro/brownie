import torch
from torch import nn
from .blocks import block_initializer_normal, SelfAttentionBlock, FourierProjection, ScalingLayer, ResidualBlock_He2015

class UNet_timecondition(nn.Module):

    def __init__(self, in_channels, U_depth = 4, U_level_blocks = 2):

        self.U_depth = U_depth
        self.U_level_blocks = self.U_level_blocks
        
        # Time embedding
        ######################################################################
        self.modules_time = nn.Sequential(
            FourierProjection(),
            block_initializer_normal(nn.Linear()),
            block_initializer_normal(nn.Linear()),
        )
        
        # Downsample arc
        ######################################################################
        self.scale_down = nn.ModuleList()
        self.modules_down = nn.ModuleList()

        for Ulevel in range(1, self.U_depth+1):
            level_modules = []
            for Ublock in range(self.U_level_blocks):
                level_modules.append(ResidualBlock_He2015())
            self.modules_down.append(nn.Sequential(
                *level_modules,
            ))
            self.scale_down.append(
                ScalingLayer("DOWN") if Ulevel < self.U_depth else nn.Identity()
            )

        # U Bottom
        ######################################################################
        self.modules_bottom = nn.Sequential(
            ResidualBlock_He2015(),
            SelfAttentionBlock(),
            ResidualBlock_He2015(),
        )
        
        # Upsample arc
        ######################################################################
        self.scale_up = nn.ModuleList()
        self.modules_up = nn.ModuleList()

        for Ulevel in range(1, self.U_depth+1):
            
            self.scale_up.append(
                ScalingLayer("UP") if Ulevel < self.U_depth else nn.Identity()
            )
            
            level_modules = []
            for Ublock in range(self.U_level_blocks):
                level_modules.append(ResidualBlock_He2015())
            self.modules_down.append(nn.Sequential(
                *level_modules,
            ))

    def forward(self, x, time):
        skip_connect = []
        state = x

        # Time embedding
        ######################################################################
        time_embedding = self.modules_time(time)

        # Downsample arc
        ######################################################################
        for Ulevel in range(self.U_depth):
            state = self.modules_down[Ulevel](state)
            skip_connect.append(state)
            state = self.scale_down[Ulevel](state)
        skip_connect = skip_connect[::-1] # reverse

        # U Bottom
        ######################################################################
        state = self.modules_bottom(state)

        # Upsample arc
        ######################################################################
        for Ulevel in range(self.U_depth):
            state = self.scale_up[Ulevel](state)
            state = self.modules_up[Ulevel](torch.cat((skip_connect[Ulevel], state)))
        
        return state
