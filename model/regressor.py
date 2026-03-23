import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cuda_cond_mul.cond_mul import CondMul
import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import random
import os

import sys 
sys.path.append("..")


class Classifier3Stage(nn.Module):
    def __init__(self,
                 height=int(704/2),
                 classes=[16, 12, 8],
                 pad=[0, 2, 4],  # pad around classes
                 ch_in=[32, 32, 32],
                 ch_in_offset=[0, 32, 64],
                 ch_latent=[[32, 32], [32, 32], [32, 32]]):
        super(Classifier3Stage, self).__init__()
        self.classes = classes
        self.pad = pad
        self.ch_in_offset = ch_in_offset
        self.ch_in = ch_in

        classes12 = classes[0] * classes[1]

        ch_latent[0].insert(0, ch_in[0])
        ch_latent[1].insert(0, ch_in[1])
        ch_latent[2].insert(0, ch_in[2])

        ch_latent[0].append(classes[0] + 2 * pad[0])
        ch_latent[1].append(classes[1] + 2 * pad[1])
        ch_latent[2].append(classes[2] + 2 * pad[2])
        self.c1 = nn.ModuleList()
        self.c2 = nn.ModuleList()
        self.c3 = nn.ModuleList()
        height = int(704/2)

        self.max_disp_mask = torch.zeros(16, height, 640)
        offset = 288.0 / 2.0
        
        gap = (640 + offset) / 16.0
        for i in range(0, 640):
            tmp = int((i + offset) // gap)
            if tmp == 0:
                self.max_disp_mask[0:tmp+1, :, i] = 1
            if tmp == 1:
                self.max_disp_mask[0:tmp+1, :, i] = 1
            if tmp == 2:
                self.max_disp_mask[0:tmp+1, :, i] = 1
            if tmp >= 3:
                self.max_disp_mask[tmp-3:tmp+1, :, i] = 1
        
        self.c1.append(CondMul(height * 16, 64, 32))
        self.c1.append(CondMul(height * 16, 32, 16))
        self.c1.append(CondMul(height * 16, 16, 16))
        
        for i in range(0, len(ch_latent[1]) - 1):
            self.c2.append(CondMul(height * classes[0], ch_latent[1][i], ch_latent[1][i + 1]))

        for i in range(0, len(ch_latent[2]) - 1):
            self.c3.append(CondMul(height * classes12, ch_latent[2][i], ch_latent[2][i + 1]))

    def cdf(self, A):
        cdf_A = torch.cumsum(A, dim=1)
        return cdf_A
    
    def get_mean_weights(self):
        mean_weights = {}
        accu = 0
        for c in self.c1:
            accu += c.weight.abs().mean()
        mean_weights["mean_weight_c1"] = accu
        accu = 0
        for c in self.c2:
            accu += c.w.abs().mean()
        mean_weights["mean_weight_c2"] = accu
        accu = 0
        for c in self.c3:
            accu += c.w.abs().mean()
        mean_weights["mean_weight_c3"] = accu
        return mean_weights

    def get_ch_slice(self, x_in, stage):
        if hasattr(self, 'ch_in_offset'):
            x = x_in[:, self.ch_in_offset[stage]:self.ch_in_offset[stage]+ self.ch_in[stage], :, :]
        else:
            # TODO: REMOVE THIS BRANCH AS SOON AS WE DON'T NEED THE LEGACY TRAINING DATA ANYMORE!
            if stage == 2:
                if self.close_far_separation:
                    # the second half of channelss dedicated to close features
                    x = x_in[:, int(x_in.shape[1] / 2):, :, :]
                else:
                    x = x_in
            elif stage == 0:
                if self.close_far_separation:
                    # the first half of channels is dedicated to high-level features
                    x = x_in[:, :int(x_in.shape[1] / 2), :, :]
                else:
                    x = x_in
            else:
                x = x_in

        # convert from (b, c, h, w) to (b, h, w, c) to (b * h * w, c)
        return x #x.permute((0, 2, 3, 1)).reshape((-1, x.shape[1])).contiguous()


    def forward(self, x_in, inds_gt=None, msk=None, output_entropies=False):        
        bs = x_in.shape[0]  # batch size
        height = x_in.shape[2]
        width = x_in.shape[3]
        device = x_in.device

        max_disp_mask = self.max_disp_mask.unsqueeze(0).repeat(bs, 1, 1, 1).to(device)
        offsets = torch.arange(0, height, device=device, dtype=torch.int32).reshape((1, 1, height, 1))

        classes12 = self.classes[0] * self.classes[1]
        classes123 = classes12 * self.classes[2]
        classes23 = self.classes[1] * self.classes[2]
        entropies = []
        

        # STEP 1:
        x = self.get_ch_slice(x_in, 0) #x = self.get_far_segment(x_in)

        # convert from (b, c, h, w) to (b, h, c, w) to (b, h*c, 1, w)
        # x = x.permute((0, 2, 1, 3)).reshape((bs, -1, 1, width))
        x_tmp = x.permute((0, 2, 3, 1)).reshape((-1, x.shape[1])).contiguous()
        x = x_tmp
        
        inds0_l = torch.zeros(bs, 1, height, width, device=device).type(torch.int32) + 1 * offsets
        inds0_l = inds0_l.flatten()


        for i in range(0, len(self.c1)):
            x = self.c1[i](x, inds0_l)
            if i < len(self.c1) - 1:
                x = F.leaky_relu(x)
                
        # convert from (b, h*c, 1, w) to (b, h, c, w) to (b, c, h, w)
        x = x.reshape((bs, height, width, -1)).permute((0, 3, 1, 2))
        x = x * max_disp_mask
        x1 = x

        inds1 = x.argmax(dim=1).unsqueeze(1)
        
        inds1_l = inds1.type(torch.int32) + self.classes[0] * offsets# add offset for each line!
        
        if output_entropies:
            p = F.softmax(x, dim=1)
            entropies.append(torch.sum(-torch.log(p) * p, dim=1).reshape((bs, 1, height, width)))

        # STEP 2:
        # convert from (b, 1, h, w) to (b * h * w)
        inds1_l = inds1_l.flatten()

        x = self.get_ch_slice(x_in, 1)
        
        # convert from (b, c, h, w) to (b, h, w, c) to (b * h * w, c)
        x_2 = x.permute((0, 2, 3, 1)).reshape((-1, x.shape[1])).contiguous()
        x = x_2
        # x_2 = x_tmp
        # x = x_2
        
        # torch.cuda.synchronize()
        # stage2_start_time = time.time()
        
        # c2
        for i in range(0, len(self.c2)):
            x = self.c2[i](x, inds1_l)
            if i < len(self.c2) - 1:
                x = F.leaky_relu(x)
            

        # (b * h * w, c) to (b * h * w, 1) to (b, 1, h, w)
        inds2 = x.argmax(dim=1).reshape((bs, 1, height, width))
        
        inds12 = inds1 * self.classes[1] + (inds2 - self.pad[1])
        inds12_l = inds12.clamp(0, classes12 - 1) + classes12 * offsets
        inds12_l_scaled_lines = inds12.clamp(0, classes12 - 1) + classes12 * (offsets)

        # (b, 1, h, w) to (b * h * w)
        inds12_l = inds12_l.type(torch.int32).flatten()
        inds12_l_scaled_lines = inds12_l_scaled_lines.type(torch.int32).flatten()
 
        if output_entropies:
            p = F.softmax(x, dim=1)
            entropies.append(torch.sum(-torch.log(p) * p, dim=1).reshape((bs, 1, height, width)))

        # STEP 3:
        x = self.get_ch_slice(x_in, 2)
        
        # convert from (b, c, h, w) to (b, h, w, c) to (b * h * w, c)
        x_3 = x.permute((0, 2, 3, 1)).reshape((-1, x.shape[1])).contiguous()
        x = x_3
        # x_3 = x_tmp
        # x = x_3
        
        # torch.cuda.synchronize()
        # stage3_start_time = time.time()
        
        # c3
        for i in range(0, len(self.c3)):
            if i < len(self.c3) - 1:
                x = F.leaky_relu(self.c3[i](x, inds12_l_scaled_lines))
            else:
                x = self.c3[i](x, inds12_l)


        # (b * h * w, c) to (b * h * w, 1) to (b, 1, h, w)
        inds3 = x.argmax(dim=1).reshape((bs, 1, height, width))

        if output_entropies:
            p = F.softmax(x, dim=1)
            entropies.append(torch.sum(-torch.log(p) * p, dim=1).reshape((bs, 1, height, width)))

        inds123_real = inds12 * self.classes[2] + (inds3 - self.pad[2])
        inds123_real = inds123_real.clamp(0, classes123 - 1) # due to padding the clamping might be necessary
        if inds_gt is None:
            if output_entropies:
                return inds123_real, entropies[0], entropies[1], entropies[2]
            return inds123_real
        else:
            
            losses = []
            inds_gt = inds_gt.clamp(0, classes123 - 1)
            inds1_gt = inds_gt // classes23
            
            loss = F.cross_entropy(x1, inds1_gt.squeeze(1).type(torch.int64), reduction='none') * 2

            losses.append(loss)
            inds1_gt = inds1_gt
            # also select the neighbouring superclasses
            loss_sum = 0
            for i in [-1, 0, 1]:
                # calculate the index of this class/ its neighbours
                inds1_l = inds1_gt + i
                inds1_l = inds1_l.clamp(0, self.classes[0] - 1)
                # calculate the local groundtruth index
                inds2_gt = inds_gt // self.classes[2] - inds1_l * self.classes[1]
                inds2_gt = inds2_gt + self.pad[1]

                # the mask masks out where this would not yield any valid samples
                mask = torch.logical_and(inds2_gt >= 0, inds2_gt < (self.classes[1] + 2 * self.pad[1]))
                inds2_gt = inds2_gt.clamp(0, self.classes[1] + 2 * self.pad[1] - 1).squeeze(1).type(torch.int64)

                inds1_l = (inds1_l + self.classes[0] * offsets).flatten().type(torch.int32)
                x = x_2
                for j in range(0, len(self.c2)):
                    x = self.c2[j](x, inds1_l)
                    if j < len(self.c2) - 1: # new j old i
                        x = F.leaky_relu(x)

                # from (b * h * w, c) to (b, h, w, c) to (b, c, h, w)
                x = x.reshape((bs, height, width, -1)).permute((0, 3, 1, 2))
                loss = F.cross_entropy(x, inds2_gt, reduction='none').unsqueeze(1) * mask
                
                loss_sum += loss

            losses.append(loss_sum)

            inds12_gt = inds_gt // self.classes[2]
            inds12_gt = inds12_gt
            loss_sum = 0
            for i in [-1, 0, 1]:
                record = i
                # calculate the index of this class/ its neighbours
                inds12_l = inds12_gt + i
                inds12_l = inds12_l.clamp(0, classes12 - 1)
                # calculate the local groundtruth index
                inds3_gt = inds_gt - inds12_l * self.classes[2]
                inds3_gt = inds3_gt + self.pad[2]

                # the mask masks out where this does not yield any valid samples
                mask = torch.logical_and(inds3_gt >= 0, inds3_gt < (self.classes[2] + 2 * self.pad[2]))
                inds3_gt = inds3_gt.clamp(0, self.classes[2] + 2 * self.pad[2] - 1).squeeze(1).type(torch.int64)

                inds12_l = (inds12_l + classes12 * offsets).type(torch.int32).flatten()
                inds12_l_scaled_lines = inds12.clamp(0, classes12 - 1) + classes12 * (offsets)
                inds12_l_scaled_lines = inds12_l_scaled_lines.type(torch.int32).flatten()
                
                x = x_3
                for i in range(0, len(self.c3)):
                    if i < len(self.c3) - 1:
                        x = F.leaky_relu(self.c3[i](x, inds12_l_scaled_lines))
                    else:
                        x = self.c3[i](x, inds12_l)

                # from (b * h * w, c) to (b, h, w, c) to (b, c, h, w)
                x = x.reshape((bs, height, width, -1)).permute((0, 3, 1, 2))
                loss = F.cross_entropy(x, inds3_gt, reduction='none').unsqueeze(1) * mask
                loss_sum += loss

            losses.append(loss_sum)
            return inds123_real, losses


# more modern regressor that allows for regressors of different resolution
class Regressor(nn.Module):

    # default parameters are the same as for CR8_reg_cond_mul_2
    def __init__(self,
                 height=448,
                 classes=[16, 12, 10],
                 class_pad=[0, 2, 3],
                 class_ch_in_offset=[0, 32, 64],
                 class_ch_in=[32, 32, 32],
                 class_ch=[[32, 32], [32, 32], [32, 32]],
                 reg_ch_in_offset=96,
                 reg_ch_in=32,
                 reg_ch=[32],
                 reg_superclasses=384,
                 reg_shared_over_lines=1,
                 reg_overlap_neighbours=1, # take channels
                 reg_pad=0.1):
        super(Regressor, self).__init__()

        classes123 = classes[0] * classes[1] * classes[2]
        self.classes = classes
        self.height = height
        self.superclasses = reg_superclasses
        self.class_factor = int(classes123 / reg_superclasses)
        self.regress_neighbours = reg_overlap_neighbours
        self.reg_line_div = reg_shared_over_lines
        self.pad_proj = reg_pad
        self.reg_ch_in_offset = reg_ch_in_offset
        self.reg_ch_in = reg_ch_in

        self.c = Classifier3Stage(height=height,
                                  classes=classes,
                                  pad=class_pad,  # pad around classes
                                  ch_in=class_ch_in,
                                  ch_in_offset=class_ch_in_offset,
                                  ch_latent=class_ch)

        self.regressor = nn.ModuleList()

        chs = [reg_ch_in] + reg_ch
        for i in range(1, len(chs) ):
            self.regressor.append(
                CondMul(int(height / self.reg_line_div) * reg_superclasses, chs[i - 1], chs[i]))
            
        self.regressor.append(CondMul(height * classes123, chs[-1], 1))


    def regress_at_leaf(self, x, inds, inds_super):
        for i in range(len(self.regressor)-1):
            x = F.leaky_relu(self.regressor[i](x, inds_super))

        return self.regressor[-1](x, inds)

    def forward(self, x_in, x_gt=None, msk=None, output_entropies=False):

        #scale the groundtruth that -pad_proj to 1.0 + pad_proj
        #are scaled to between 0.0 and 1.0
        if x_gt is not None:
            x_gt = x_gt * (1.0 - 2.0 * self.pad_proj) + self.pad_proj

        height = x_in.shape[2]
        width = x_in.shape[3]
        classes123 = self.classes[0] * self.classes[1] * self.classes[2]
        bs = x_in.shape[0]

        device = x_in.device
        if hasattr(self, 'reg_data_start'):
            self.reg_ch_in_offset = self.reg_data_start

        if hasattr(self, 'reg_ch_in_offset'):
            x_for_r = x_in[:, self.reg_ch_in_offset:self.reg_ch_in_offset + self.reg_ch_in]
        else:
            if self.c.close_far_separation:
                # the second half of channelss dedicated to close features
                x_for_r = x_in[:, int(x_in.shape[1] / 2):, :, :]
            else:
                x_for_r = x_in

        # convert from (b, c, h, w) to (b, h, w, c) to (b * h * w, c)
        x_for_r = x_for_r.permute((0, 2, 3, 1)).reshape((-1, x_for_r.shape[1])).contiguous()

        # reshape from (b, c, h, w) to (b, h, c, w) to (b, h * c, 1, w)
        x_in = x_in.permute((0, 2, 1, 3)).reshape((bs, -1, 1, width)) # todo: maybe don't overwrite x_in here!
        # the first stage is to adapt to features to something that has meaning on this line!
        x = x_in
        x_l = x
        #convert from (b, h * c, 1, w) to (b, h, c, w) to (b, c, h, w)
        x_l = x_l.reshape((bs, height, -1, width)).permute((0, 2, 1, 3))

        # create vector with index offsets along the vertical dimension (1, 1, h, 0)
        line_offsets = torch.arange(0, height, device=device).unsqueeze(1).unsqueeze(0).unsqueeze(0)
        if x_gt is None:

            # the input for the classifier (as well as the output) should come in (b, c, h, w)
            if output_entropies:
                inds, entropy1, entropy2, entropy3 = self.c(x_l, output_entropies=True)
            else:
                inds = self.c(x_l, output_entropies=False)
            inds_super = inds // self.class_factor + self.superclasses * (line_offsets // self.reg_line_div)
            inds_l = inds + line_offsets * classes123
            inds_super = inds_super.flatten().type(torch.int32)
            inds_l = inds_l.flatten().type(torch.int32)

            x_l = x_for_r

            r = self.regress_at_leaf(x_l, inds_l, inds_super).flatten()

            x = (inds.flatten().type(torch.float32) + r) * (1.0 / float(classes123))
            x_real = x.reshape((bs, 1, height, width))


            #undo the scaling that is made at the beginning
            x_real = (x_real - self.pad_proj) / (1.0 - 2.0 * self.pad_proj)

            if output_entropies:
                return x_real, entropy1, entropy2, entropy3
            return x_real
        else:

            inds_gt = (x_gt * classes123).type(torch.int32).clamp(0, classes123 - 1)
            inds, class_losses = self.c(x_l, inds_gt, msk)

            #x = F.leaky_relu(self.r1(x_in))
            # from (b, h * c, 1, w) to (b, h, c, w) to (b * h * w, c)
            #x_l = x_in.reshape((bs, height, -1, width)).permute((0, 1, 3, 2))
            #x_l = x_l.reshape((bs * height * width, -1)).contiguous()
            x_l = x_for_r

            # calculate the regression only x
            x_reg_combined = torch.zeros((bs, 1 + 2 * self.regress_neighbours, height, width),
                                         device=device)
            for offset in range(-self.regress_neighbours, self.regress_neighbours+1):
                inds_gt = (inds_gt + offset).clamp(0, classes123 - 1)
                inds_super = inds_gt // self.class_factor
                inds_super = inds_super + self.superclasses * (line_offsets // self.reg_line_div)
                inds_super = inds_super.flatten().type(torch.int32)
                inds_l = (inds_gt + classes123 * line_offsets).flatten().type(torch.int32)

                r = self.regress_at_leaf(x_l, inds_l, inds_super).reshape((bs, 1, height, width))

                #r = self.r2(x_l, inds_gt).flatten()#todo:remove this reactivate the two lines above
                x_reg = (inds_gt.type(torch.float32) + r) * (1.0 / float(classes123))
                x_reg = x_reg.reshape((bs, 1, height, width))
                x_reg_combined[:, [offset+self.regress_neighbours], :, :] = x_reg

            # calculate the real x
            inds_super = inds // self.class_factor
            inds_super = inds_super + self.superclasses * (line_offsets // self.reg_line_div)
            inds_super = inds_super.flatten().type(torch.int32)
            inds_l = (inds + classes123 * line_offsets).flatten().type(torch.int32)
            # STEP 1
            #x = F.leaky_relu(self.r2(x_l, inds_super))
            # STEP 3 + reshape
            # from (b * h * w, 1) to (b, 1, h, w)
            #r = self.r3(x, inds_l).reshape((bs, 1, height, width))
            r = self.regress_at_leaf(x_l, inds_l, inds_super).reshape((bs, 1, height, width))

            x = (inds.type(torch.float32) + r) * (1.0 / float(classes123))
            x_real = x.reshape((bs, 1, height, width))


            #undo the scaling that is made at the beginning of this function
            x_real = (x_real - self.pad_proj) / (1.0 - 2.0 * self.pad_proj)
            x_reg_combined = (x_reg_combined - self.pad_proj) / (1.0 - 2.0 * self.pad_proj)
            return x_reg_combined, class_losses, x_real, None


class RegressorLinewise(nn.Module):
    def __init__(self,
                 lines=448,
                 ch_in=128,
                 ch_latent=[64, 64, 128, 1024, 1024, 128]):
        super(RegressorLinewise, self).__init__()

        chs = [ch_in] + ch_latent
        self.seq = nn.Sequential()
        for ch_in, ch_out, i in zip(chs[:-1], chs[1:], range(len(chs)-1)):
            self.seq.add_module(f"conv{i}", nn.Conv2d(ch_in*lines, ch_out*lines, 1, groups=lines))
            self.seq.add_module(f"bn{i}", nn.BatchNorm2d(ch_out * lines))
            self.seq.add_module(f"ReLU{i}", nn.LeakyReLU())
        self.seq.add_module(f"conv_out", nn.Conv2d(chs[-1] * lines, lines, 1, groups=lines))

    def forward(self, x, x_gt=None, output_entropies=False):
        bs = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        # convert from (b, c, h, w) to (b, h, c, w) to (b, h*c, 1, w)
        x = x.permute((0, 2, 1, 3)).reshape((bs, -1, 1, width))
        x = self.seq(x)
        # convert from (b, h*c, 1, w) to (b, h, c, w) to (b, c, h, w)
        x = x.reshape(bs, height, -1, width).permute((0, 2, 1, 3))

        if x_gt != None:
            return x, {}, x, {}
        return x

class RegressorNone(nn.Module):
    def __init__(self):
        super(RegressorNone, self).__init__()

    def forward(self, x, x_gt=None, output_entropies=False):
        if x_gt != None:
            return x, {}, x, {}
        return x