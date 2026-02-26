import logging

import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from VBMF import VBMF

from pytorch_utils import *

log = logging.getLogger(__name__)

# Returns tuned ranks according to the desired compression factor
def choose_compression(layer, ranks, compression_factor=2, flag='Tucker2'):
    '''
    Compute tuned ranks according to the desired compression 
    factor. Sometimes VBMF returns too large ranks; hence the 
    decomposition makes the layer bigger instead of shrinking it. 
    This function prevents it. 
    N.B. by default, if the compression is higher than 2
    the ranks will be untouched.

    Args:
        layer: the layer to be compressed
        ranks : estimated ranks 
        compression_factor: how much the layer will compressed
        flag: string, choose compression over different decompositions. 
              default is Tucker. 
    Returns: 
        the newly estimated rank according to desired compression
    '''
    # PyTorch format is [OUT, IN, k1, k2]
    weights = layer.weight.data.cpu().numpy()
    T = weights.shape[0]
    S = weights.shape[1]
    d = weights.shape[2]

    if flag == 'Tucker2':
        compression = ((d**2) *S* T) / ((S*ranks[0] + ranks[0]*ranks[1] * (d**2) + T*ranks[1]) )
        ranks[0] = ranks[0] * 3
        log.debug('initial compression: %s', compression)
    
        # compression must be 2 or more, otherwise arbitrary ranks will be chosen!
        if compression <= 2:
            while compression <= compression_factor:
                ranks[0] = ranks[0] // 2
                ranks[1] = ranks[1] // 2
                if ranks[0] < 1 or ranks[1] < 1:
                    ranks[0] = max(1, ranks[0])
                    ranks[1] = max(1, ranks[1])
                    break
                compression = ((d**2) * S * T) / ((S * ranks[0] + ranks[0] * ranks[1] * (d**2) + T * ranks[1]))

        log.debug('compression factor for layer %s: %s', weights.shape, compression)

    elif flag == 'cpd':
        rank = ranks[0] # it is a single value
        compression = ((d**2)*T*S) / (rank*(S+2*d+T))
        if compression <= compression_factor:
            rank = ((d**2) * S * T) / (compression_factor * (S +2*d+ T))
            ranks[0] = np.floor(rank).astype(int) 

            compression_factor = ((d**2) * S * T) / (rank * (S +2*d+ T))
            log.debug('compression factor for layer %s: %s', weights.shape, compression_factor)

        else:
            log.debug('compression factor for layer %s: %s', weights.shape, compression)
    else:
        raise NotImplementedError('Decomposition not yet supported: %s' % flag)

    return ranks


def estimate_ranks(layer):
    """
    Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """
    weights = layer.weight.data.cpu().numpy()
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]

    ranks = choose_compression(
        layer, ranks, compression_factor=30, flag='Tucker2')

    return ranks


def cp_ranks(layer):
    weights = layer.weight.data.cpu().numpy()
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    rank = max(diag_0.shape[0], diag_1.shape[1])
    log.debug('VBMF estimated rank: %d', rank)
    ranks = [rank, rank]
  #  rank, _=choose_compression(
  #      layer, ranks, compression_factor=30, flag='cpd')
    return rank


# TODO: v0.2 SVD support
# SVD_weights, FC_SVD_compression, conv1x1_SVD_compression removed — planned for v0.2


def cp_decomposition_conv_layer(layer, rank, matlab=False):
    """ Gets a conv layer and a target rank, di
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensor.
    X = layer.weight.data.cpu().numpy()
    size = max(X.shape)

    if matlab:
        last, first, vertical, horizontal = load_cpd_weights('dumps/TODO.mat')

    else:
        if size >= 256:
            _, factors = parafac(X, rank=rank, init='random')
        else:
            _, factors = parafac(X, rank=rank, init='svd')
        last, first, vertical, horizontal = factors

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                             out_channels=first.shape[1],
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
                                               out_channels=vertical.shape[1],
                                               kernel_size=(
                                                   vertical.shape[0], 1),
                                               stride=layer.stride,
                                               padding=(layer.padding[0], 0),
                                               dilation=layer.dilation,
                                               groups=vertical.shape[1],
                                               bias=False)

    depthwise_horizontal_layer = torch.nn.Conv2d(in_channels=horizontal.shape[1],
                                                 out_channels=horizontal.shape[1],
                                                 kernel_size=(
                                                     1, horizontal.shape[0]),
                                                 stride=layer.stride,
                                                 padding=(0, layer.padding[0]),
                                                 dilation=layer.dilation,
                                                 groups=horizontal.shape[1],
                                                 bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                             out_channels=last.shape[0],
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=True)
    pointwise_r_to_t_layer.bias.data = layer.bias.data

    depthwise_vertical_layer_weights = np.expand_dims(np.expand_dims(
        vertical.transpose(1, 0), axis=1), axis=-1)
    depthwise_horizontal_layer_weights = np.expand_dims(np.expand_dims(
        horizontal.transpose(1, 0), axis=1), axis=1)
    pointwise_s_to_r_layer_weights = np.expand_dims(
        np.expand_dims(first.transpose(1, 0), axis=-1), axis=-1)
    pointwise_r_to_t_layer_weights = np.expand_dims(np.expand_dims(
        last, axis=-1), axis=-1)

    set_layer_weights(depthwise_horizontal_layer,
                      depthwise_horizontal_layer_weights)
    set_layer_weights(depthwise_vertical_layer,
                      depthwise_vertical_layer_weights)
    set_layer_weights(pointwise_s_to_r_layer,
                      pointwise_s_to_r_layer_weights)
    set_layer_weights(pointwise_r_to_t_layer,
                      pointwise_r_to_t_layer_weights)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer,
                  depthwise_horizontal_layer, pointwise_r_to_t_layer]
    return nn.Sequential(*new_layers)


def cp_decomposition_conv_layer_BN(layer, rank, matlab=False):
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensor.
    X = layer.weight.data.cpu().numpy()
    size = max(X.shape)

    if matlab:
        last, first, vertical, horizontal = load_cpd_weights(
            'dumps/TODO.mat')

    else:
        if size >= 256:
            _, factors = parafac(X, rank=rank, init='random')
        else:
            _, factors = parafac(X, rank=rank, init='svd')
        last, first, vertical, horizontal = factors

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                             out_channels=first.shape[1],
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1],
                                               out_channels=vertical.shape[1],
                                               kernel_size=(
                                                   vertical.shape[0], 1),
                                               stride=layer.stride,
                                               padding=(layer.padding[0], 0),
                                               dilation=layer.dilation,
                                               groups=vertical.shape[1],
                                               bias=False)

    depthwise_horizontal_layer = torch.nn.Conv2d(in_channels=horizontal.shape[1],
                                                 out_channels=horizontal.shape[1],
                                                 kernel_size=(
                                                     1, horizontal.shape[0]),
                                                 stride=layer.stride,
                                                 padding=(0, layer.padding[0]),
                                                 dilation=layer.dilation,
                                                 groups=horizontal.shape[1],
                                                 bias=False)

    add_bias = layer.bias is not None

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                             out_channels=last.shape[0],
                                             kernel_size=1,
                                             stride=layer.stride,
                                             padding=0,
                                             dilation=layer.dilation,
                                             bias=add_bias)
    if add_bias:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    # Transpose dimensions back to what PyTorch expects
    depthwise_vertical_layer_weights = np.expand_dims(np.expand_dims(
        vertical.transpose(1, 0), axis=1), axis=-1)
    depthwise_horizontal_layer_weights = np.expand_dims(np.expand_dims(
        horizontal.transpose(1, 0), axis=1), axis=1)
    pointwise_s_to_r_layer_weights = np.expand_dims(
        np.expand_dims(first.transpose(1, 0), axis=-1), axis=-1)
    pointwise_r_to_t_layer_weights = np.expand_dims(np.expand_dims(
        last, axis=-1), axis=-1)

    # Fill in the weights of the new layers
    depthwise_horizontal_layer.weight.data = \
        torch.from_numpy(np.float32(depthwise_horizontal_layer_weights))
    depthwise_vertical_layer.weight.data = \
        torch.from_numpy(np.float32(depthwise_vertical_layer_weights))
    pointwise_s_to_r_layer.weight.data = \
        torch.from_numpy(np.float32(pointwise_s_to_r_layer_weights))
    pointwise_r_to_t_layer.weight.data = \
        torch.from_numpy(np.float32(pointwise_r_to_t_layer_weights))

    # create BatchNorm layers wrt to decomposed layers weights
    bn_first = nn.BatchNorm2d(first.shape[1])
    bn_vertical = nn.BatchNorm2d(vertical.shape[1])
    bn_horizontal = nn.BatchNorm2d(horizontal.shape[1])
    bn_last = nn.BatchNorm2d(last.shape[0])

    new_layers = [pointwise_s_to_r_layer, bn_first, depthwise_vertical_layer, bn_vertical,
                  depthwise_horizontal_layer, bn_horizontal,  pointwise_r_to_t_layer,
                  bn_last]
    return nn.Sequential(*new_layers)


def tucker_decomposition_conv_layer(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """

    ranks = estimate_ranks(layer)
    (core, [last, first]), _errs = partial_tucker(
        layer.weight.data.cpu().numpy(), modes=[0, 1], rank=ranks, init='svd')

    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                  out_channels=first.shape[1],
                                  kernel_size=1,
                                  stride=layer.stride,
                                  padding=0,
                                  dilation=layer.dilation,
                                  bias=False)

    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                                 out_channels=core.shape[0],
                                 kernel_size=layer.kernel_size,
                                 stride=layer.stride,
                                 padding=layer.padding,
                                 dilation=layer.dilation,
                                 bias=False)

    last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                 out_channels=last.shape[0],
                                 kernel_size=1,
                                 stride=layer.stride,
                                 padding=0,
                                 dilation=layer.dilation,
                                 bias=True)

    last_layer.bias.data = layer.bias.data

    first = first.transpose((1, 0))
    first_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(first.copy(), axis=-1), axis=-1)))
    last_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(last.copy(), axis=-1), axis=-1)))
    core_layer.weight.data = torch.from_numpy(np.float32(core.copy()))

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def tucker_decomposition_conv_layer_BN(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """

    ranks = estimate_ranks(layer)
    (core, [last, first]), _errs = partial_tucker(
        layer.weight.data.cpu().numpy(), modes=[0, 1], rank=ranks, init='svd')

    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                  out_channels=first.shape[1],
                                  kernel_size=1,
                                  stride=layer.stride,
                                  padding=0,
                                  dilation=layer.dilation,
                                  bias=False)

    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                                 out_channels=core.shape[0],
                                 kernel_size=layer.kernel_size,
                                 stride=layer.stride,
                                 padding=layer.padding,
                                 dilation=layer.dilation,
                                 bias=False)

    last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                 out_channels=last.shape[0],
                                 kernel_size=1,
                                 stride=layer.stride,
                                 padding=0,
                                 dilation=layer.dilation,
                                 bias=True)

    last_layer.bias.data = layer.bias.data

    bn_first = nn.BatchNorm2d(first.shape[1])
    bn_core = nn.BatchNorm2d(core.shape[0])
    bn_last = nn.BatchNorm2d(last.shape[0])

    first = first.transpose((1, 0))
    first_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(first.copy(), axis=-1), axis=-1)))
    last_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(last.copy(), axis=-1), axis=-1)))
    core_layer.weight.data = torch.from_numpy(np.float32(core.copy()))

    new_layers = [first_layer, bn_first,
                  core_layer, bn_core, last_layer, bn_last]
    return nn.Sequential(*new_layers)



# tucker_xavier, cp_xavier_conv_layer removed — superseded by use_xavier_init config flag
