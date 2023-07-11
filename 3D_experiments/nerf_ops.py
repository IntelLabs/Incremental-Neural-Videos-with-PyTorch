# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:10:50 2022

@author: asupikov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import nerf definitions
from run_nerf_helpers import NeRF

# copy weights from linear layer to linear layer
def copy_lin(dst, src) :
    dst.weight.copy_(src.weight)
    dst.bias.copy_(dst.bias)
    return dst

def diff_lin(dst, src1, src0) :
    # dst = src1-src0
    dst.weight = src1.weight - src0.weight
    dst.bias = src1.bias - src0.bias
    return dst

def add_lin(dst, src1, src0) :
    # dst = src1-src0
    dst.weight = src1.weight + src0.weight
    dst.bias = src1.bias + src0.bias
    return dst

# function to copy weights from one network to another
# TODO: eventually parse parameters()
def copy_layers(dst, src) :
    if isinstance(src, NeRF) and isinstance(dst, NeRF) :
        for l in range(len(src.pts_linears)):
            copy_lin(dst.pts_linears[l], src.pts_linears[l])
        for l in range(len(src.views_linears)):
            copy_lin(dst.views_linears[l], src.views_linears[l])
        copy_lin(dst.feature_linear, src.feature_linear)
        copy_lin(dst.alpha_linear, src.alpha_linear)
        copy_lin(dst.rgb_linear, src.rgb_linear)
    else:
        assert False, f"Unknown NN type or types dont match. dst:{type(dst)}, src:{type(src)}" 
    return dst

def diff_layers(dst, src1, src0) :
    # dst = src1-src0
    if isinstance(src1, NeRF) and isinstance(src0, NeRF) and isinstance(dst, NeRF) :
        for l in range(len(src1.pts_linears)):
            diff_lin(dst.pts_linears[l], src1.pts_linears[l], src0.pts_linears[l])
        for l in range(len(src1.views_linears)):
            diff_lin(dst.views_linears[l], src1.views_linears[l], src0.views_linears[l])
        diff_lin(dst.feature_linear, src1.feature_linear, src0.feature_linear)
        diff_lin(dst.alpha_linear, src1.alpha_linear, src0.alpha_linear)
        diff_lin(dst.rgb_linear, src1.rgb_linear, src0.rgb_linear)
    else:
        assert False, f"Unknown NN type or types dont match. dst:{type(dst)}, src1:{type(src1)}, src0:{type(src0)}" 
    return dst

def add_layers(dst, src1, src0) :
    # dst = src1+src0
    if isinstance(src1, NeRF) and isinstance(src0, NeRF) and isinstance(dst, NeRF) :
        for l in range(len(src1.pts_linears)):
            add_lin(dst.pts_linears[l], src1.pts_linears[l], src0.pts_linears[l])
        for l in range(len(src1.views_linears)):
            add_lin(dst.views_linears[l], src1.views_linears[l], src0.views_linears[l])
        add_lin(dst.feature_linear, src1.feature_linear, src0.feature_linear)
        add_lin(dst.alpha_linear, src1.alpha_linear, src0.alpha_linear)
        add_lin(dst.rgb_linear, src1.rgb_linear, src0.rgb_linear)
    else:
        assert False, f"Unknown NN type or types dont match. dst:{type(dst)}, src1:{type(src1)}, src0:{type(src0)}" 
    return dst

# motion/structure layers ops    
def copy_motion_layers(dst, src, first_l) :
    if isinstance(src, NeRF) and isinstance(dst, NeRF) :
        for l in range(min(first_l, len(src.pts_linears))):
            copy_lin(dst.pts_linears[l], src.pts_linears[l])
        #for l in range(len(src.views_linears)):
        #    copy_lin(dst.views_linears[l], src.views_linears[l])
        #copy_lin(dst.feature_linear, src.feature_linear)
        #copy_lin(dst.alpha_linear, src.alpha_linear)
        #copy_lin(dst.rgb_linear, src.rgb_linear)
    else:
        assert False, f"Unknown NN type or types dont match. dst:{type(dst)}, src:{type(src)}" 
    return dst

def copy_non_motion_layers(dst, src, first_l) :
    if isinstance(src, NeRF) and isinstance(dst, NeRF) :
        for l in range(first_l, len(src.pts_linears)):
            copy_lin(dst.pts_linears[l], src.pts_linears[l])
        for l in range(len(src.views_linears)):
            copy_lin(dst.views_linears[l], src.views_linears[l])
        copy_lin(dst.feature_linear, src.feature_linear)
        copy_lin(dst.alpha_linear, src.alpha_linear)
        copy_lin(dst.rgb_linear, src.rgb_linear)
    else:
        assert False, f"Unknown NN type or types dont match. dst:{type(dst)}, src:{type(src)}" 
    return dst

def diff_motion_layers(dst, src1, src0, first_l) :
    # dst = src1-src0
    if isinstance(src1, NeRF) and isinstance(src0, NeRF) and isinstance(dst, NeRF) :
        for l in range(min(first_l, len(src1.pts_linears))):
            diff_lin(dst.pts_linears[l], src1.pts_linears[l], src0.pts_linears[l])
    else:
        assert False, f"Unknown NN type or types dont match. dst:{type(dst)}, src1:{type(src1)}, src0:{type(src0)}" 
    return dst
    
def add_motion_layers(dst, src1, src0, first_l) :
    # dst = src1+src0
    if isinstance(src1, NeRF) and isinstance(src0, NeRF) and isinstance(dst, NeRF) :
        for l in range(min(first_l, len(src1.pts_linears))):
            add_lin(dst.pts_linears[l], src1.pts_linears[l], src0.pts_linears[l])
    else:
        assert False, f"Unknown NN type or types dont match. dst:{type(dst)}, src1:{type(src1)}, src0:{type(src0)}" 
    return dst

def freeze_last_layers(dst, start_l, freeze = True) :
    # freeze all network layers starting freeze
    # dst - NeRF to use, 
    # start_l  - 0-based index of first layer to freeze
    # freeze - True - freeze, False - unfreeze
    grad_flag = not freeze
    if isinstance(dst, NeRF) :
        for l in range(len(dst.pts_linears)):
            if l < start_l:
                # print(f"layer {l} not frozen")
                continue
            dst.pts_linears[l].weight.requires_grad = grad_flag
            dst.pts_linears[l].bias.requires_grad = grad_flag
        for l in range(len(dst.views_linears)):
            dst.views_linears[l].weight.requires_grad = grad_flag
            dst.views_linears[l].bias.requires_grad = grad_flag
        dst.feature_linear.weight.requires_grad = grad_flag
        dst.feature_linear.bias.requires_grad = grad_flag
        dst.alpha_linear.weight.requires_grad = grad_flag
        dst.alpha_linear.bias.requires_grad = grad_flag
        dst.rgb_linear.weight.requires_grad = grad_flag
        dst.rgb_linear.bias.requires_grad = grad_flag
    else:
        assert False, f"Unknown NN type or types dont match. dst:{type(dst)}" 
    return dst
