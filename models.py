import torch
from torch import nn
import math
from typing import Union, Sequence, Tuple
from utils import define_act_layer
from swintransformer import SwinTransformer, look_up_option
from einops import rearrange

import os 
from medpy.io import load, header
import numpy as np


class MultiTaskModel(nn.Module):
    def __init__(self, task, in_features, hidden_units=None, act_layer=nn.ReLU(), dropout=0.25) -> None:

        super().__init__()
        self.act = act_layer
        incoming_features = in_features
        hidden_layer_list = []
        self.task = task
        for hidden_unit in hidden_units:
            hidden_block = nn.Sequential(
                nn.Linear(incoming_features, hidden_unit),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(hidden_unit),
                nn.Dropout(dropout),
            )
            hidden_layer_list.append(hidden_block)
            incoming_features = hidden_unit
        self.hidden_layer = nn.Sequential(*hidden_layer_list)
       
        out_features = 2 if self.task=="multitask" else 1
        self.classifier = nn.Linear(hidden_units[-1], out_features)
        self.output_act = nn.Sigmoid()
        

    def forward(self, x):
        x = self.hidden_layer(x)
        # print(x.shape)
        classifier = self.classifier(x)
        # print(classifier)
        if self.task =="multitask":
            grade, hazard = self.output_act(classifier)[:, 0], self.output_act(classifier)[:, 1]
            
            return grade, hazard
        else:
            # print(self.output_act(classifier))
            return self.output_act(classifier)
            return classifier


class SelfAttentionBi(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SelfAttentionBi, self).__init__()

        self.WQ = nn.Linear(dim_in, dim_out)
        self.WK = nn.Linear(dim_in, dim_out)
        self.WV = nn.Linear(dim_in, dim_out)
        self.root = math.sqrt(dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mod1, mod2):
        x = torch.stack((mod1, mod2), dim=1)
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        QK = torch.bmm(Q, K.transpose(1, 2))
        attention_matrix = self.softmax(QK/self.root)
        out = torch.bmm(attention_matrix, V)
        return out


class FusionModelBi(nn.Module):
    def __init__(self, args, dim_in, dim_out):
        super(FusionModelBi, self).__init__()
        self.fusion_type = args.fusion_type
        act_layer = define_act_layer(args.act_type)

        if self.fusion_type == "attention":
            self.attention_module = SelfAttentionBi(dim_in, dim_out)
            self.taskmodel = MultiTaskModel(
                args.task, dim_out*2, args.hidden_units, act_layer, args.dropout)
        elif self.fusion_type == "fused_attention":
            self.attention_module = SelfAttentionBi(dim_in, dim_out)
            self.taskmodel = MultiTaskModel(
                args.task, (dim_out+1)**2, args.hidden_units, act_layer, args.dropout)
        elif self.fusion_type == "kronecker":
            self.taskmodel = MultiTaskModel(
                args.task, (dim_in+1)**2, args.hidden_units, act_layer, args.dropout)
        elif self.fusion_type == "concatenation":
            self.taskmodel = MultiTaskModel(
                args.task, dim_in*2, args.hidden_units, act_layer, args.dropout)
        else:
            raise NotImplementedError(
                f'Fusion method {self.fusion_type} is not implemented')

    def forward(self, vec1, vec2):

        if self.fusion_type == "attention":
            x = self.attention_module(vec1, vec2)
            x = x.view(x.shape[0], x.shape[1]*x.shape[2])

        elif self.fusion_type == "kronecker":
            vec1 = torch.cat(
                (vec1, torch.ones((vec1.shape[0], 1)).to(vec1.device)), 1)
            vec2 = torch.cat(
                (vec2, torch.ones((vec2.shape[0], 1)).to(vec2.device)), 1)
            x = torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1)).flatten(
                start_dim=1)

        elif self.fusion_type == "fused_attention":
            vec1, vec2 = self.attention_module(
                vec1, vec2)[:, 0, :], self.attention_module(vec1, vec2)[:, 1, :]
            vec1 = torch.cat(
                (vec1, torch.ones((vec1.shape[0], 1)).to(vec1.device)), 1)
            vec2 = torch.cat(
                (vec2, torch.ones((vec2.shape[0], 1)).to(vec2.device)), 1)
            x = torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1)).flatten(
                start_dim=1)

        elif self.fusion_type == "concatenation":
            x = torch.cat((vec1, vec2), dim=1)

        else:
            raise NotImplementedError(
                f'Fusion method {self.fusion_type} is not implemented')
        return self.taskmodel(x)
    
class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SelfAttention, self).__init__()

        self.WQ = nn.Linear(dim_in, dim_out)
        self.WK = nn.Linear(dim_in, dim_out)
        self.WV = nn.Linear(dim_in, dim_out)
        self.root = math.sqrt(dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mod1, mod2, mod3):
        x = torch.stack((mod1, mod2 ,mod3), dim=1)
        Q = self.WQ(x) 
        K = self.WK(x) 
        V = self.WV(x) 

        QK = torch.bmm(Q, K.transpose(1, 2)) 
        attention_matrix = self.softmax(QK/self.root)
        out = torch.bmm(attention_matrix, V)
        return out

    
class FusionModel(nn.Module):
    def __init__(self, args, dim_in, dim_out):
        super(FusionModel, self).__init__()
        self.fusion_type = args.fusion_type
        act_layer = define_act_layer(args.act_type)
    
        if self.fusion_type == "attention":
            self.attention_module = SelfAttention(dim_in, dim_out)
            self.taskmodel = MultiTaskModel(
                args.task, dim_out*3, args.hidden_units, act_layer, args.dropout)
            
        elif self.fusion_type == "fused_attention":
            self.attention_module = SelfAttention(dim_in, dim_out)
            self.taskmodel = MultiTaskModel(
                args.task, (dim_out+1)**3, args.hidden_units, act_layer, args.dropout)
            
        elif self.fusion_type == "kronecker":
            self.taskmodel = MultiTaskModel(
                args.task, (dim_in+1)**3, args.hidden_units, act_layer, args.dropout)
            
        elif self.fusion_type == "concatenation":
            self.taskmodel = MultiTaskModel(
                args.task, dim_in*3, args.hidden_units, act_layer, args.dropout)
            
        else:
            raise NotImplementedError(
                f'Fusion method {self.fusion_type} is not implemented')
        
    def forward(self, vec1, vec2, vec3):
        
        if self.fusion_type == "attention":
            x = self.attention_module(vec1, vec2, vec3)
            x = x.view(x.shape[0], x.shape[1]*x.shape[2])
            
        elif self.fusion_type == "kronecker":
            vec1 = torch.cat(
                (vec1, torch.ones((vec1.shape[0], 1)).to(vec1.device)), 1)
            vec2 = torch.cat(
                (vec2, torch.ones((vec2.shape[0], 1)).to(vec2.device)), 1)
            vec3 = torch.cat(
                (vec3, torch.ones((vec3.shape[0], 1)).to(vec3.device)), 1)
            x12 = torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1)).flatten(
            start_dim=1)
            x = torch.bmm(x12.unsqueeze(2), vec3.unsqueeze(1)).flatten(
                start_dim=1)
            
        elif self.fusion_type == "fused_attention":
            vec1, vec2, vec3 = self.attention_module(
                vec1, vec2, vec3)[:, 0, :], self.attention_module(vec1, vec2, vec3)[:, 1, :] , self.attention_module(vec1, vec2, vec3)[:, 2, :]
            vec1 = torch.cat(
                (vec1, torch.ones((vec1.shape[0], 1)).to(vec1.device)), 1)
            vec2 = torch.cat(
                (vec2, torch.ones((vec2.shape[0], 1)).to(vec2.device)), 1)
            vec3 = torch.cat(
                (vec3, torch.ones((vec3.shape[0], 1)).to(vec3.device)), 1)
            x12 = torch.bmm(vec1.unsqueeze(2), vec2.unsqueeze(1)).flatten(
                start_dim=1)
            x = torch.bmm(x12.unsqueeze(2), vec3.unsqueeze(1)).flatten(
                start_dim=1)
            # print(x.shape)
            
        elif self.fusion_type == "concatenation":
            x = torch.cat((vec1, vec2, vec3), dim=1)

        else: 
            raise NotImplementedError(
                f'Fusion method {self.fusion_type} is not implemented')
        return self.taskmodel(x)


class SwinTransformerRadiologyModel(nn.Module):

    def __init__(
        self,
        patch_size: Union[Sequence[int], int],
        window_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
    ) -> None:
        """
        Input requirement : [BxCxDxHxW]
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).


        """

        super().__init__()

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample) if isinstance(
                downsample, str) else downsample,
        )
        self.norm = nn.LayerNorm(feature_size*4)
        self.avgpool = nn.AdaptiveAvgPool3d([1, 1, 1])
        self.dim_reduction = nn.Conv3d(feature_size*4, out_channels, 1)
        
    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        hidden_output = rearrange(
            hidden_states_out[2], "b c d h w -> b d h w c")
        nomalized_hidden_states_out = self.norm(hidden_output)
        nomalized_hidden_states_out = rearrange(
            nomalized_hidden_states_out, "b d h w c -> b c d h w")
        output = self.avgpool(nomalized_hidden_states_out)
        output = torch.flatten(self.dim_reduction(output), 1)

        
        return output
    

class SwinTransformerPathologyModel(nn.Module):

    def __init__(
        self,
        patch_size: Union[Sequence[int], int],
        window_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
        downsample="merging",
    ) -> None:
        """
        Input requirement : [BxCxDxHxW]
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).


        """

        super().__init__()

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.normalize = normalize
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample) if isinstance(
                downsample, str) else downsample,
        )
        self.feature_size = feature_size
        self.norm = nn.LayerNorm(self.feature_size*(2**(len(num_heads))))
        self.avgpool = nn.AdaptiveAvgPool3d([1, 1, 1])
        
        self.dim_reduction = nn.Conv3d(self.feature_size*4, out_channels, 1)

    def forward(self, x_in, batch_size):
        hidden_states_out = self.swinViT(x_in, self.normalize)
        hidden_output = rearrange(
            hidden_states_out[-1], "b c h w -> b h w c")
        # print(f"shape after rearrange: {hidden_output.shape}")
        nomalized_hidden_states_out = self.norm(hidden_output)
        nomalized_hidden_states_out = rearrange(
            nomalized_hidden_states_out, "b h w c -> b c h w")
        # print(
        #     f"shape after rearrange and norm: {nomalized_hidden_states_out.shape}")
        output = self.avgpool(torch.reshape(nomalized_hidden_states_out, (batch_size, self.feature_size*4, -1, nomalized_hidden_states_out.shape[-2], nomalized_hidden_states_out.shape[-1])))
        # print(
        #      f"shape after average pooling: {output.shape}")
        output = torch.flatten(self.dim_reduction(output), 1)
        # print(
        #      f"shape after flatten: {output.shape}")
        return output
    

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.extractor_ct_tumor = SwinTransformerRadiologyModel(
            patch_size=(1, 2, 2),
            window_size=[[4, 4, 4], [4, 4, 4]],
            in_channels=4,
            out_channels=args.feature_size,
            depths=(2, 2),
            num_heads=(3, 6),
            feature_size=int(args.feature_size/2),
            norm_name="instance",
            drop_rate=0,
            attn_drop_rate=0,
            dropout_path_rate=0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3
        )
        self.extractor_ct_lymph = SwinTransformerRadiologyModel(
            patch_size=(1, 2, 2),
            window_size=[[4, 4, 4], [4, 4, 4]],
            in_channels=4,
            out_channels=args.feature_size,
            depths=(2, 2),
            num_heads=(3, 6),
            feature_size=int(args.feature_size/2),
            norm_name="instance",
            drop_rate=0,
            attn_drop_rate=0,
            dropout_path_rate=0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3
        )
        self.extractor_pathology = SwinTransformerPathologyModel(
            patch_size = (2,2),
            window_size =[[4,4],[4,4]],
            in_channels=192,
            out_channels = args.feature_size,
            depths = (2, 2),
            num_heads = (3,6),
            feature_size = int(args.feature_size/2),
            norm_name = "instance",
            drop_rate= 0,   ########
            attn_drop_rate = 0,  ########
            dropout_path_rate = 0,
            normalize = True,
            use_checkpoint = False,
            spatial_dims=2
            )
        self.fusion = FusionModel(args, args.feature_size, args.dim_out)
    
        
    def forward(self, ct_tumor, ct_lymph, path, batch_size):
        features_tumor = self.extractor_ct_tumor(ct_tumor)
        # print("features_tumor", features_tumor.shape)
        features_lymph = self.extractor_ct_tumor(ct_lymph)
        # print("features_lymph", features_lymph.shape)
        features_pathology = self.extractor_pathology(path, batch_size)
        # print("features_pathology", features_pathology.shape)
        
        output = self.fusion(
            features_tumor, features_lymph, features_pathology)
        #print("output shape", output.shape)
        return output
        




class RModel(nn.Module):
    def __init__(self, args):
        super(RModel, self).__init__()
        self.extractor_ct_tumor = SwinTransformerRadiologyModel(
            patch_size=(1, 2, 2),
            window_size=[[4, 4, 4], [4, 4, 4], [8, 8, 8], [4, 4, 4]],
            in_channels=4,
            out_channels=args.feature_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=int(args.feature_size/2),
            norm_name="instance",
            drop_rate=0,
            attn_drop_rate=0,
            dropout_path_rate=0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3
        )
        self.extractor_ct_lymph = SwinTransformerRadiologyModel(
            patch_size=(1, 2, 2),
            window_size=[[4, 4, 4], [4, 4, 4], [8, 8, 8], [4, 4, 4]],
            in_channels=4,
            out_channels=args.feature_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=int(args.feature_size/2),
            norm_name="instance",
            drop_rate=0,
            attn_drop_rate=0,
            dropout_path_rate=0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3
        )
        self.fusion = FusionModelBi(args, args.feature_size, args.dim_out)
        
    def forward(self, ct_tumor, ct_lymph):
        features_tumor = self.extractor_ct_tumor(ct_tumor)
        features_lymph = self.extractor_ct_tumor(ct_lymph)
        
        output = self.fusion(
            features_tumor, features_lymph)
        ##print("output shape", output.shape)
        return output





class PModel(nn.Module):
    def __init__(self, args):
        super(PModel, self).__init__()
        self.extractor_pathology = SwinTransformerPathologyModel(
            patch_size = (2,2),
            window_size =[[4,4],[4,4],[8,8],[4,4]],
            in_channels=192,
            out_channels = args.feature_size,
            depths = (2, 2, 2, 2),
            num_heads = (3,6,12, 24),
            feature_size = int(args.feature_size/2),
            norm_name = "instance",
            drop_rate= 0.2,   ########
            attn_drop_rate = 0.2,  ########
            dropout_path_rate = 0,
            normalize = True,
            use_checkpoint = False,
            spatial_dims=2
            )
        
        self.fusion = FusionModelBi(args, args.feature_size, args.dim_out)
        # self.fusion = MultiTaskModel(
        #         args.task, args.dim_out, args.hidden_units, act_layer=nn.ReLU(), dropout=0)
    
        
    def forward(self, path, batch_size):
        
        features_pathology = self.extractor_pathology(path, batch_size)
        output = self.fusion(features_pathology, features_pathology)
        # print("output shape", output.shape)
        return output
    


