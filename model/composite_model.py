import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from model.backboneSliced import *
from model.regressor import *
import sys
import time
from common.common import LCN_tensors

import sys 
sys.path.append("..")


class CompositeModel(nn.Module):
    def __init__(self, backbone, regressor, half_precision=False, apply_lcn=False):
        super(CompositeModel, self).__init__()
        self.half_precision = half_precision
    
        self.backbone = proposed_light_Lut()
        self.regressor = regressor

    def forward(self, x, x_gt=None, msk=None, output_entropies=False, measure_time=False, lut_5 = None, w5=None):
        if self.apply_lcn:
            lcn, _, _ = LCN_tensors(x)
            x = torch.cat((x, lcn), 1)

        if x_gt is not None:
            if self.half_precision:
                with autocast():
                    x, debugs = self.backbone(x, True)
                x = x.type(torch.float32)
            else:
                # x, debugs = self.backbone(x, True)
                x = self.backbone(x,lut_5 = lut_5, w5 = w5)

            results = self.regressor(x, x_gt, msk)

            # for key, val in debugs.items():
            #     results[-1][key] = val

            return results
        else:
            if measure_time:
                self.half_precision = False
                tsince = ((time.time() * 1000))
            if self.half_precision:
                with autocast():
                    x = self.backbone(x)
                x = x.type(torch.float32)
            else:
                x = self.backbone(x,lut_5 = lut_5, w5 = w5)
                # return x
                # x_2_5_0, x_2_5_0_lut = self.backbone(x,lut_5 = lut_5, w5 = w5)
                # return x_2_5_0, x_2_5_0_lut
            if measure_time:
                torch.cuda.synchronize()
                ttime_elapsed = (time.time() * 1000) - tsince
                print(f"backbone time elapsed {ttime_elapsed} ms")

                tsince = int(round(time.time() * 1000))
            results = self.regressor(x, x_gt,
                                     output_entropies=output_entropies)
            if measure_time:
                torch.cuda.synchronize()
                ttime_elapsed = (time.time() * 1000) - tsince
                print(f"regressor time elapsed {ttime_elapsed} ms")

            return results


def GetModel(args, config):
    if config["training"]["load_model"] != "":
        model = torch.load(config["training"]["load_model"])
        model.eval()
        return model

    in_channels = 1
    if args.LCN:
        in_channels = 2

    # generating one of 3 regressors
    regressor_name = config["regressor"]["name"]
    if regressor_name == "Regressor":
        regressor = Regressor(height=config["regressor"]["lines"],
                              classes=config["regressor"]["classes"],
                              class_pad=config["regressor"]["class_padding"],
                              class_ch_in_offset=config["regressor"]["class_ch_in_offset"],
                              class_ch_in=config["regressor"]["class_ch_in"],
                              class_ch=config["regressor"]["class_ch"],
                              reg_ch_in_offset=config["regressor"]["reg_data_start"],
                              reg_ch_in=config["regressor"]["reg_ch_in"],
                              reg_ch=config["regressor"]["reg_ch"],
                              reg_superclasses=config["regressor"]["reg_superclasses"],
                              reg_overlap_neighbours=config["regressor"]["reg_overlap_neighbours"],  # take channels
                              reg_shared_over_lines=config["regressor"]["reg_shared_over_lines"],
                              reg_pad=config["regressor"]["reg_pad"])
    elif regressor_name == "None":
        regressor = RegressorNone()
    elif regressor_name == "Lines":
        regressor = RegressorLinewise(lines=config["regressor"]["lines"],
                                      ch_in=config["regressor"]["ch_in"],
                                      ch_latent=config["regressor"]["ch_latent"])
    else:
        print(f"{regressor_name} does not name a valid regressor.", file=sys.stderr)
        sys.exit(1)

    backbone_name = config["backbone"]["name"].replace("Sliced", "")
    if backbone_name == "Backbone":
        backboneType = BackboneSlice
        constructor = lambda pad, channels, downsample: BackboneSlice(
            channels=config["backbone"]["channels"],
            kernel_sizes=config["backbone"]["kernel_sizes"],
            channels_sub=config["backbone"]["channels_sub"],
            kernel_sizes_sub=config["backbone"]["kernel_sizes_sub"],
            use_bn=True,
            pad=pad, channels_in=channels)
    elif backbone_name == "UNet":
        backboneType = UNet
        constructor = lambda pad, channels, downsample: UNet(
            cin=config["backbone"]["input_channels"],
            cout=config["backbone"]["output_channels"],
            half_out=True,
            channel_size_scale=config["backbone"]["unet_channel_count_scale"]
        )
    else:
        print(f"{backbone_name} does not name a valid backbone.", file=sys.stderr)
        sys.exit(1)

    if config["backbone"]["name"].endswith("Sliced"):
        backbone = BackboneSlicer(backboneType, constructor,
                                  config["backbone"]["slices"],
                                  in_channels=in_channels,
                                  downsample_output=args.downsample_output)
    else:
        backbone = constructor('both', in_channels, args.downsample_output)

    model = CompositeModel(backbone, regressor, args.half_precision, apply_lcn=args.LCN)
    return model
