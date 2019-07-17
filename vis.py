import sys
import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torchvision import models

from region_loss import RegionLoss
from cfg import *
from misc_functions import preprocess_image, recreate_image


class CNNLayerVisualization():
    def __init__(self, model, selected_layer, selected_filter):
            self.model = model
            self.model.eval()
            self.selected_layer = selected_layer
            self.selected_filter = selected_filter
            self.conv_output = 0
            # Generate a random image
            self.created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
            # Create the folder to export images if not exists
            if not os.path.exists('./generated'):
                os.makedirs('./generated')


    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
            #print(self.conv_output)
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)


    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        #self.hook_layer()
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image)
        # Define optimizer for the image
        optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        """
        learning_rate = 0.001
        batch_size = 16
        momentum = 0.9
        decay = 0.0005
        optimizer = optim.SGD([self.processed_image], lr=learning_rate/batch_size, 
                               momentum=momentum, dampening=0, weight_decay=decay*batch_size)
        """

        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
               
                x = layer(x)

                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break

            self.conv_output = x[0, self.selected_filter]
            #print(self.conv_output)
            #print(self.conv_output)
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)

            # loss
            #loss = region_loss(output, target)


            #print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image
            
            if i % 30 == 0:
                cv2.imwrite('./generated/layer_vis_l' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) +'.jpg', self.created_image)
            


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x


class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x


def get_seq(blocks):
    seq = []
    prev_filters = 3
    out_filters =[]

    for block in blocks:
        if block['type'] == 'net':
            prev_filters = int(block['channels'])
            continue

        elif block['type'] == 'convolutional':
            batch_normalize = int(block['batch_normalize'])
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)//2 if is_pad else 0
            activation = block['activation']

            if batch_normalize:
                seq.append(nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                seq.append(nn.BatchNorm2d(filters))
            else:
                seq.append(nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))

            if activation == 'leaky':
                seq.append(nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'relu':
                seq.append(nn.ReLU(inplace=True))

            prev_filters = filters
            out_filters.append(prev_filters)

        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            if stride > 1:
                model = nn.MaxPool2d(pool_size, stride)
            else:
                model = MaxPoolStride1()
            out_filters.append(prev_filters)
            seq.append(model)


        elif block['type'] == 'route':
            sequential = nn.Sequential(*seq)
            return sequential


def create_network(blocks):
    models = nn.ModuleList()

    prev_filters = 3
    out_filters =[]
    conv_id = 0
    for block in blocks:
        if block['type'] == 'net':
            prev_filters = int(block['channels'])
            continue
        elif block['type'] == 'convolutional':
            conv_id = conv_id + 1
            batch_normalize = int(block['batch_normalize'])
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)//2 if is_pad else 0
            activation = block['activation']
            model = nn.Sequential()
            if batch_normalize:
                model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
            else:
                model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
            if activation == 'leaky':
                model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'relu':
                model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
            prev_filters = filters
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            if stride > 1:
                model = nn.MaxPool2d(pool_size, stride)
            else:
                model = MaxPoolStride1()
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'avgpool':
            model = GlobalAvgPool2d()
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'softmax':
            model = nn.Softmax()
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'cost':
            if block['_type'] == 'sse':
                model = nn.MSELoss(size_average=True)
            elif block['_type'] == 'L1':
                model = nn.L1Loss(size_average=True)
            elif block['_type'] == 'smooth':
                model = nn.SmoothL1Loss(size_average=True)
            out_filters.append(1)
            models.append(model)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            prev_filters = stride * stride * prev_filters
            out_filters.append(prev_filters)
            models.append(Reorg(stride))
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            ind = len(models)
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            if len(layers) == 1:
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                assert(layers[0] == ind - 1)
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_filters.append(prev_filters)
            models.append(EmptyModule())
        elif block['type'] == 'shortcut':
            ind = len(models)
            prev_filters = out_filters[ind-1]
            out_filters.append(prev_filters)
            models.append(EmptyModule())
        elif block['type'] == 'connected':
            filters = int(block['output'])
            if block['activation'] == 'linear':
                model = nn.Linear(prev_filters, filters)
            elif block['activation'] == 'leaky':
                model = nn.Sequential(
                           nn.Linear(prev_filters, filters),
                           nn.LeakyReLU(0.1, inplace=True))
            elif block['activation'] == 'relu':
                model = nn.Sequential(
                           nn.Linear(prev_filters, filters),
                           nn.ReLU(inplace=True))
            prev_filters = filters
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'region':
            loss = RegionLoss()
            anchors = block['anchors'].split(',')
            loss.anchors = [float(i) for i in anchors]
            loss.num_classes = int(block['classes'])
            loss.num_anchors = int(block['num'])
            loss.anchor_step = len(loss.anchors)/loss.num_anchors
            loss.object_scale = float(block['object_scale'])
            loss.noobject_scale = float(block['noobject_scale'])
            loss.class_scale = float(block['class_scale'])
            loss.coord_scale = float(block['coord_scale'])
            out_filters.append(prev_filters)
            models.append(loss)
        else:
            print('unknown type %s' % (block['type']))

    return models


def load_weights(weightfile, blocks, models):
    fp = open(weightfile, 'rb')
    header = np.fromfile(fp, count=4, dtype=np.int32)
    #self.header = torch.from_numpy(header)
    #self.seen = self.header[3]
    buf = np.fromfile(fp, dtype = np.float32)
    fp.close()

    start = 0
    ind = -2
    for block in blocks:
        if start >= buf.size:
            break
        ind = ind + 1
        if block['type'] == 'net':
            continue
        elif block['type'] == 'convolutional':
            model = models[ind]
            batch_normalize = int(block['batch_normalize'])
            if batch_normalize:
                start = load_conv_bn(buf, start, model[0], model[1])
            else:
                start = load_conv(buf, start, model[0])
        elif block['type'] == 'connected':
            model = models[ind]
            if block['activation'] != 'linear':
                start = load_fc(buf, start, model[0])
            else:
                start = load_fc(buf, start, model)


if __name__ == '__main__':
    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    
    blocks = parse_cfg(cfgfile)
    pretrained_model = get_seq(blocks)
    print(pretrained_model)

    modelss = create_network(blocks)
    load_weights(weightfile, blocks, modelss)

    #pretrained_model = models.vgg16(pretrained=True).features

    for cnn_layer in range(0, len(pretrained_model)):
        if type(pretrained_model[cnn_layer]) == torch.nn.modules.conv.Conv2d:
            for filter_pos in range(0, pretrained_model[cnn_layer].out_channels):
                print("layer: ", cnn_layer)
                print(pretrained_model[cnn_layer], filter_pos)

                layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)
                layer_vis.visualise_layer_with_hooks()

        else:
            layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, 0)
            layer_vis.visualise_layer_with_hooks()  
    
    """
    cnn_layer = 14
    filter_pos = 5


    # Fully connected layer is not needed
    #pretrained_model = models.vgg16(pretrained=True).features
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()
    """
