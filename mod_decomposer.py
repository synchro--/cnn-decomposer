# For now, it is only a set of functions.
#   The ranks are estimated with a Python implementation of VBMF
#   https://github.com/CasvandenBogaard/VBMF
#
# Frameworks supported:
# - pytorch
# - keras   (todo!)
# - TF      (todo!)
#
#
#

import logging

import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from VBMF import VBMF
import collections
from pytorch_utils import *

log = logging.getLogger(__name__)

"""
Decomposer module to abstract tensor decomposition methods for
Convolutional Neural Networks.

Major Features
-----------
- TD block configuration
- CP decomposition
- Tucker decomposition
- arbitrary compression ratio
- arbitrary rank
- VBMF rank estimation
"""


### Public methods ###
######################

def estimate_tucker_ranks(layer, compression_factor=0):
    """
    Unfold the 2 modes of the specified layer tensor, on which the decomposition
    will be performed, and estimates the ranks of the matrices using VBMF.
    Args:
        layer: (torch.nn.Conv2d) the layer that will be decomposed
        compression_factor: (double or int) preferred compression factor to be enforced over
                            the rank estimation of VBMF, i.e. if VBMF estimated ranks
                            are too high for the desired compression, they will be
                            iteratively divided by a constant factor K until they reach compression rate Cr >= desired compression rate.
    Returns:
        ranks: (list) the estimated ranks [R3, R4]
    """
    weights = layer.weight.data.cpu().numpy()
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    log.debug('VBMF estimated ranks: %s', ranks)

    if compression_factor:
        # Check if the VBMF ranks are small enough
        ranks = _choose_compression(layer, ranks,
                                    compression_factor, flag='Tucker2')
    return ranks


def estimate_cp_ranks(layer, compression_factor=0):
    """
    Unfold the 2 modes of the specified layer tensor, on which the decomposition
    will be performed, and estimates the ranks of the matrices using VBMF
    Args:
        layer: (torch.nn.Conv2d) the layer that will be decomposed
        compression_factor: (double or int) preferred compression factor to be enforced over
                            the rank estimation of VBMF
    Returns:
        rank: (int) the estimated rank R
    """
    weights = layer.weight.data.cpu().numpy()
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    #_, diag_2, _, _ = VBMF.EVBMF(unfold_2)
    rank = max(diag_0.shape[0], diag_1.shape[1])
    log.debug('VBMF estimated rank (max): %d', rank)
    if rank == 0:
        rank = 10
    ranks = [rank, rank]

    # if specified, choose desired compression
    if compression_factor:
        rank, _ = _choose_compression(layer, ranks,
                                      compression_factor, flag='cpd')
    return rank

def pytorch_cp_layer_decomposition(layer, compression=0, offline=False, filename=''):
    """
        Gets a conv layer and a target rank, and returns a nn.Sequential
        compressed CNN.
        Args:
            layer: (torch.nn.Conv2d) the conv layer to decompose
            compression: desired compression factor for the CP-decomposition. This parameter
            will influence the value of the rank of the decomposition. At default it seeks the best
            tradeoff between compression and accuracy
            offline: bool, if true the weights will be loaded from the file
                        specified in file name
            filename: string, file from which we have to load the weights.

        Returns:
            The compressed CNN model.
    """
    rank = estimate_cp_ranks(layer, compression)
    new_layers = _cp_decomposition(layer, rank, offline, filename)
    return nn.Sequential(*new_layers)


def pytorch_cp_layer_decomposition_BN(layer, rank=0, offline=False, filename=''):
    """
        Gets a conv layer and a target rank, returns a nn.Sequential compressed CNN,
        with BN units between the compressed layers.
        Args:
            layer: the conv layer to decompose
            rank: the rank of the CP-decomposition
            offline: bool, if true the weights will be loaded from the file
                        specified in file name
            filename: string, file from which we have to load the weights.

        Returns:
            The compressed CNN model.
    """
    (first_pointwise, separable_vertical,
     separable_horizontal, last_pointwise) = _cp_decomposition(
        layer, rank, offline, filename)

    bn_first = nn.BatchNorm2d(first_pointwise.out_channels)
    bn_vertical = nn.BatchNorm2d(separable_vertical.out_channels)
    bn_horizontal = nn.BatchNorm2d(separable_horizontal.out_channels)
    bn_last = nn.BatchNorm2d(last_pointwise.out_channels)

    new_layers = [first_pointwise, bn_first, separable_vertical, bn_vertical,
                  separable_horizontal, bn_horizontal,  last_pointwise,
                  bn_last]
    return nn.Sequential(*new_layers)


def pytorch_tucker_layer_decomposition(layer, compression=0, offline=False, filename=''):
    """
        Gets a conv layer and a target rank, and returns a nn.Sequential
        compressed CNN.
        Args:
            layer: the conv layer to decompose
            rank: the rank of the CP-decomposition
            offline: bool, if true the weights will be loaded from the file
                        specified in file name
            filename: string, file from which we have to load the weights.

        Returns:
            The compressed CNN model.
    """
    ranks = estimate_tucker_ranks(layer, compression_factor=compression)
    new_layers = _tucker_layer_decomposition(layer, ranks)
    return nn.Sequential(*new_layers)



# TODO: v0.2 SVD support
# FC_SVD_compression, conv1x1_SVD_compression removed — planned for v0.2


##################### private utils #######################################
###########################################################################


def _choose_compression(layer, ranks, compression_factor=2, flag='Tucker2', framework='pytorch'):
    """
    Compute tuned ranks according to the desired compression
    factor. Sometimes VBMF returns too large ranks; hence the
    decomposition makes the layer bigger instead of shrinking it.
    This function prevents it.

    N.B. by default, if the compression is higher than 2
    the ranks selected by VBMF will be untouched.


    Args:
        layer: the layer to be compressed
        ranks : estimated ranks
        compression_factor: how much the layer will compressed
        flag: string, choose compression over different decompositions.
                default is Tucker.
        framework: keywoard specifiying a framework between pytorch, keras and tensorflow.

    Returns:
        the newly estimated rank according to desired compression
    """

    if framework == 'pytorch':
        # PyTorch format is [OUT, IN, k1, k2]
        weights = layer.weight.data.cpu().numpy()
        T = weights.shape[0]
        S = weights.shape[1]
        d = weights.shape[2]

    # if Tucker2 is the selected method, then we compute the correspondent compression factor
    # and then diminish the two ranks R3,R4 iteratively until we reach the desired compression
    if flag == 'Tucker2':
        compression = ((d**2)*S*T) / \
            ((S*ranks[0] + ranks[0]*ranks[1] * (d**2) + T*ranks[1]))
        ranks[0] = ranks[0] * 3
        log.debug('initial compression: %s', compression)

        if compression <= 2:
            while compression <= compression_factor:
                ranks[0] = ranks[0] // 2
                ranks[1] = ranks[1] // 2
                if ranks[0] < 1 or ranks[1] < 1:
                    ranks[0] = max(1, ranks[0])
                    ranks[1] = max(1, ranks[1])
                    break
                compression = ((d**2) * S * T) / ((S * ranks[0] +
                                                   ranks[0] * ranks[1] * (d**2) + T * ranks[1]))

        log.debug('compression factor for layer %s: %s', weights.shape, compression)

    elif flag == 'cpd':
        rank = ranks[0]  # it is a single value
        compression = ((d**2)*T*S) / (rank*(S+2*d+T))
        if compression <= compression_factor:
            rank = ((d**2) * S * T) / (compression_factor * (S + 2*d + T))
            ranks[0] = np.floor(rank).astype(int)

            compression_factor = ((d**2) * S * T) / (rank * (S + 2*d + T))
            log.debug('compression factor for layer %s: %s', weights.shape, compression_factor)

        else:
            log.debug('compression factor for layer %s: %s', weights.shape, compression)
    else:
        raise NotImplementedError('Decomposition not yet supported: %s' % flag)

    return ranks


def _SVD_weights(weights, t):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.

    Parameters:
        W: N x M weights matrix
        t: number of singular values to retain
    Returns:
        Ul, L: matrices such that W \approx Ul*L
    """

    # numpy doesn't seem to have a fast truncated SVD algorithm...
    # this could be faster
    U, s, V = np.linalg.svd(weights, full_matrices=False)

    U = U[:, :t]
    Sigma = s[:t]
    Vt = V[:t, :]

    L = np.dot(np.diag(Sigma), Vt)
    return U, L


def _cp_decomposition(layer, rank, offline=False, filename=''):
    """ Gets a conv layer and a target rank, di
        returns a nn.Sequential object with the decomposition
        Args:
            layer: the conv layer to decompose
            rank: the rank of the CP-decomposition
            offline: bool, if true the weights will be loaded from the file
                        specified in file name
            filename: string, file from which we have to load the weights.

        Returns:
            The compressed 4 layers that substitutes the original one.
    """

    # Perform CP decomposition on the layer weight tensor.
    log.debug('computing CP-decomposition of layer %s with rank %d', layer, rank)
    X = layer.weight.data.cpu().numpy()
    size = max(X.shape)

    # THIS SHOULD BE ENHANCED BY USING A SUBPROCESS CALL
    # WHICH CALLS THE MATLAB SCRIPT AND RETRIEVE THE RESULT
    # TODO
    if offline:
        last, first, vertical, horizontal = load_cpd_weights(filename)

    else:
        # SVD init leads to generally better overall compression.
        # However, it can stall for large matrices.
        if size >= 256:
            _, factors = parafac(X, rank=rank, init='random')
        else:
            _, factors = parafac(X, rank=rank, init='svd')
        last, first, vertical, horizontal = factors

    first_pointwise = torch.nn.Conv2d(in_channels=first.shape[0],
                                      out_channels=first.shape[1],
                                      kernel_size=1,
                                      stride=layer.stride,
                                      padding=0,
                                      dilation=layer.dilation,
                                      bias=False)

    separable_vertical = torch.nn.Conv2d(in_channels=vertical.shape[1],
                                         out_channels=vertical.shape[1],
                                         kernel_size=(
        vertical.shape[0], 1),
        stride=layer.stride,
        padding=(layer.padding[0], 0),
        dilation=layer.dilation,
        groups=vertical.shape[1],
        bias=False)

    separable_horizontal = torch.nn.Conv2d(in_channels=horizontal.shape[1],
                                           out_channels=horizontal.shape[1],
                                           kernel_size=(
        1, horizontal.shape[0]),
        stride=layer.stride,
        padding=(0, layer.padding[0]),
        dilation=layer.dilation,
        groups=horizontal.shape[1],
        bias=False)

    last_pointwise = torch.nn.Conv2d(in_channels=last.shape[1],
                                     out_channels=last.shape[0],
                                     kernel_size=1,
                                     stride=layer.stride,
                                     padding=0,
                                     dilation=layer.dilation,
                                     bias=True)
    last_pointwise.bias.data = layer.bias.data

    # Transpose dimensions back to what PyTorch expects
    separable_vertical_weights = np.expand_dims(np.expand_dims(
        vertical.transpose(1, 0), axis=1), axis=-1)
    separable_horizontal_weights = np.expand_dims(np.expand_dims(
        horizontal.transpose(1, 0), axis=1), axis=1)
    first_pointwise_weights = np.expand_dims(
        np.expand_dims(first.transpose(1, 0), axis=-1), axis=-1)
    last_pointwise_weights = np.expand_dims(np.expand_dims(
        last, axis=-1), axis=-1)

    set_layer_weights(separable_horizontal,
                      separable_horizontal_weights)
    set_layer_weights(separable_vertical,
                      separable_vertical_weights)
    set_layer_weights(first_pointwise,
                      first_pointwise_weights)
    set_layer_weights(last_pointwise,
                      last_pointwise_weights)

    return [first_pointwise, separable_vertical, separable_horizontal, last_pointwise]


def _tucker_layer_decomposition(layer, ranks, offline=False, filename=''):
    """ Gets a conv layer and a target rank and
        returns a nn.Sequential object with the decomposition
        Args:
            layer: the conv layer to decompose
            rank: the rank of the CP-decomposition
            offline: bool, if true the weights will be loaded from the file
                        specified in file name
            filename: string, file from which we have to load the weights.

        Returns:
            The compressed 3 layers that substitutes the original one.
    """

    log.debug('computing Tucker decomposition of layer %s with ranks %d,%d',
              layer, ranks[0], ranks[1])

    # THIS SHOULD BE ENHANCED BY USING A SUBPROCESS CALL
    # WHICH CALLS THE MATLAB SCRIPT AND RETRIEVE THE RESULT
    if offline:
        last, first, vertical, horizontal = load_cpd_weights(filename)

    else:
        (core, [last, first]), _errs = partial_tucker(
            layer.weight.data.cpu().numpy(), modes=[0, 1], rank=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                  out_channels=first.shape[1],
                                  kernel_size=1,
                                  stride=layer.stride,
                                  padding=0,
                                  dilation=layer.dilation,
                                  bias=False)

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                                 out_channels=core.shape[0],
                                 kernel_size=layer.kernel_size,
                                 stride=layer.stride,
                                 padding=layer.padding,
                                 dilation=layer.dilation,
                                 bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                 out_channels=last.shape[0],
                                 kernel_size=1,
                                 stride=layer.stride,
                                 padding=0,
                                 dilation=layer.dilation,
                                 bias=True)

    last_layer.bias.data = layer.bias.data

    # Transpose add dimensions to fit into the PyTorch tensors
    first = first.transpose((1, 0))
    first_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(first.copy(), axis=-1), axis=-1)))
    last_layer.weight.data = torch.from_numpy(np.float32(
        np.expand_dims(np.expand_dims(last.copy(), axis=-1), axis=-1)))
    core_layer.weight.data = torch.from_numpy(np.float32(core.copy()))

    new_layers = [first_layer, core_layer, last_layer]
    return new_layers
