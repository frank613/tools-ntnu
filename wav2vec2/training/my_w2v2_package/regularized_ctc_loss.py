import os
import torch as T
from torch.autograd import Variable
import numpy as np
import pdb
import copy
import time

cuda = True
if cuda:
    floatX = T.cuda.FloatTensor
    intX = T.cuda.IntTensor
    byteX = T.cuda.ByteTensor
    longX = T.cuda.LongTensor
else:
    floatX = T.FloatTensor
    intX = T.IntTensor
    byteX = T.ByteTensor
    longX = T.LongTensor


def m_eye(n, k=0):
    assert k < n and k >= 0
    if k == 0:
        return T.eye(n).type(floatX)
    else:
        return T.cat((T.cat((T.zeros(n-k, k), T.eye(n-k)), dim=1), T.zeros(k, n)), dim=0).type(floatX)
    
def regularized_eye(n): ## n is the length of the matrix
    assert n%2 != 0
    index1 = T.arange(0, n, 2)
    index2 = index1
    ret = T.ones(n,n).type(floatX)
    ret[index1,index2] = 0 
    return ret

#we allow the first ane last pad to repeat, 
def regularized_eye_mid(n): ## n is the length of the matrix
    assert n%2 != 0
    index1 = T.arange(2, n-2, 2)
    index2 = index1
    ret = T.ones(n,n).type(floatX)
    ret[index1,index2] = 0 
    return ret

def ctc_loss(pred, pred_len, token, token_len, blank=0):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)
    '''
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps = 0

    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)

    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)

    # recurrence relation
    sec_diag = T.cat((T.zeros((batch, 2)).type(floatX), T.ne(token_with_blank[:, :-2], token_with_blank[:, 2:]).type(floatX)), dim=1) * T.ne(token_with_blank, blank).type(floatX)	# (batch, 2U+1)
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1) + m_eye(length, k=2).repeat(batch, 1, 1) * sec_diag[:, None, :]	# (batch, 2U+1, 2U+1)

    # alpha
    alpha_t = T.cat((pred[0, :, :2], T.zeros(batch, 2*U-1).type(floatX)), dim=1) # (batch, 2U+1)
    probability = alpha_t[None] # (1, batch, 2U+1)

    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1, Time).type(longX):
        alpha_t = T.bmm(alpha_t[:, None], recurrence_relation)[:, 0] * pred[t]
        probability = T.cat((probability, alpha_t[None]), dim=0)

    labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    labels_prob = labels_2 + labels_1

    cost = -T.log(labels_prob+eps)
    return cost

def log_batch_dot(alpha_t, rec):
    '''
    alpha_t: (batch, 2U+1)
    rec: (batch, 2U+1, 2U+1)
    '''
    eps_nan = -1e8
    # a+b
    _sum = alpha_t[:, :, None] + rec
    _max_sum = T.max(_sum, dim=1)[0]
    nz_mask1 = T.gt(_max_sum, eps_nan) # max > eps_nan
    nz_mask2 = T.gt(_sum, eps_nan)     # item > eps_nan

    # a+b-max
    _sum = _sum - _max_sum[:, None]

    # exp
    _exp = T.zeros_like(_sum).type(floatX)
    _exp[nz_mask2] = T.exp(_sum[nz_mask2])

    # sum exp
    _sum_exp = T.sum(_exp, dim=1)

    out = T.ones_like(_max_sum).type(floatX) * eps_nan
    out[nz_mask1] = T.log(_sum_exp[nz_mask1]) + _max_sum[nz_mask1]
    return out

def log_sum_exp_axis(a, uniform_mask=None, dim=0):
    assert dim == 0
    eps_nan = -1e8
    eps = 1e-26
    _max = T.max(a, dim=dim)[0]

    if not uniform_mask is None:
        nz_mask2 = T.gt(a, eps_nan) * uniform_mask
        nz_mask1 = T.gt(_max, eps_nan) * T.ge(T.max(uniform_mask, dim=dim)[0], 1)
    else:
        nz_mask2 = T.gt(a, eps_nan)
        nz_mask1 = T.gt(_max, eps_nan)

    # a-max
    a = a - _max[None]

    # exp
    _exp_a = T.zeros_like(a).type(floatX)
    _exp_a[nz_mask2] = T.exp(a[nz_mask2])

    # sum exp
    _sum_exp_a = T.sum(_exp_a, dim=dim)

    out = T.ones_like(_max).type(floatX) * eps_nan
    out[nz_mask1] = T.log(_sum_exp_a[nz_mask1] + eps) + _max[nz_mask1]
    return out

def log_sum_exp(*arrs):
#    return T.max(a.clone(), b.clone()) + T.log1p(T.exp(-T.abs(a.clone()-b.clone())))
    c = T.cat(list(map(lambda x:x[None], arrs)), dim=0)
    return log_sum_exp_axis(c, dim=0)

##regurlarize the transition from blank to blank = 0
def reg_ctc_loss_log(pred, pred_len, token, token_len, blank=0):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)
    '''
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps_nan = -1e8

    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)

    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)

    # recurrence relation
    sec_diag = T.cat((T.zeros((batch, 2)).type(floatX), T.ne(token_with_blank[:, :-2], token_with_blank[:, 2:]).type(floatX)), dim=1) * T.ne(token_with_blank, blank).type(floatX)	# (batch, 2U+1)
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1) + m_eye(length, k=2).repeat(batch, 1, 1) * sec_diag[:, None, :]	# (batch, 2U+1, 2U+1)
    # regulariztion
    reg_mat = regularized_eye(length)
    recurrence_relation = recurrence_relation * reg_mat
    recurrence_relation = eps_nan * (T.ones_like(recurrence_relation) - recurrence_relation)

    # alpha
    alpha_t = T.cat((pred[0, :, :2], T.ones(batch, 2*U-1).type(floatX)*eps_nan), dim=1) # (batch, 2U+1)
    probability = alpha_t[None] # (1, batch, 2U+1)
    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1, Time).type(longX):
        alpha_t = log_batch_dot(alpha_t, recurrence_relation) + pred[t]
        probability = T.cat((probability, alpha_t[None]), dim=0)

    labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    labels_prob = log_sum_exp(labels_2, labels_1)
#     pdb.set_trace()

    cost = -labels_prob
    return cost

##regurlarize the transition from blank to blank = 0 and nob means no jump from token to token without passing through a blank
def reg_ctc_loss_nob_log(pred, pred_len, token, token_len, blank=0):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)
    '''
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps_nan = -1e8

    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)

    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)

    # recurrence relation, no jump from token to token is allowed. 
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1)
    # regulariztion
    reg_mat = regularized_eye(length)
    recurrence_relation = recurrence_relation * reg_mat
    recurrence_relation = eps_nan * (T.ones_like(recurrence_relation) - recurrence_relation)

    # alpha
    alpha_t = T.cat((pred[0, :, :2], T.ones(batch, 2*U-1).type(floatX)*eps_nan), dim=1) # (batch, 2U+1)
    probability = alpha_t[None] # (1, batch, 2U+1)
    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1, Time).type(longX):
        alpha_t = log_batch_dot(alpha_t, recurrence_relation) + pred[t]
        probability = T.cat((probability, alpha_t[None]), dim=0)

    labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    labels_prob = log_sum_exp(labels_2, labels_1)
#     pdb.set_trace()

    cost = -labels_prob
    return cost

##same as nob version, but we allow the first and last blank repeats
def reg_ctc_loss_nob_mid_log(pred, pred_len, token, token_len, blank=0):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)
    '''
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps_nan = -1e8

    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)

    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)

    # recurrence relation, no jump from token to token is allowed. 
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1)
    # regulariztion
    reg_mat = regularized_eye_mid(length)
    recurrence_relation = recurrence_relation * reg_mat
    recurrence_relation = eps_nan * (T.ones_like(recurrence_relation) - recurrence_relation)

    # alpha
    alpha_t = T.cat((pred[0, :, :2], T.ones(batch, 2*U-1).type(floatX)*eps_nan), dim=1) # (batch, 2U+1)
    probability = alpha_t[None] # (1, batch, 2U+1)
    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1, Time).type(longX):
        alpha_t = log_batch_dot(alpha_t, recurrence_relation) + pred[t]
        probability = T.cat((probability, alpha_t[None]), dim=0)

    labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    labels_prob = log_sum_exp(labels_2, labels_1)
#     pdb.set_trace()

    cost = -labels_prob
    return cost

##regurlarize the transition from blank to blank = 0, adding SIL
def reg_ctc_loss_sil_log(pred, pred_len, token, token_len, blank=0, sil_token=-1):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)
    '''
    assert(sil_token > 0)
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps_nan = -1e8

    #token with sil:
    token_with_sil = T.cat((T.full((batch,1), sil_token).type(longX), token), dim=1) #(batch, U+1)
    token_with_sil = T.cat((token_with_sil, T.full((batch,1), blank).type(longX)), dim=1) #(batch, U+2)
    U = U + 2
    index1 = T.arange(batch)
    index2 = token_len + 1
    token_with_sil[index1, index2] = sil_token
    token = token_with_sil
    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)

    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)

    # recurrence relation
    sec_diag = T.cat((T.zeros((batch, 2)).type(floatX), T.ne(token_with_blank[:, :-2], token_with_blank[:, 2:]).type(floatX)), dim=1) * T.ne(token_with_blank, blank).type(floatX)	# (batch, 2U+1)
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1) + m_eye(length, k=2).repeat(batch, 1, 1) * sec_diag[:, None, :]	# (batch, 2U+1, 2U+1)
    # regulariztion
    reg_mat = regularized_eye(length)
    recurrence_relation = recurrence_relation * reg_mat
    recurrence_relation = eps_nan * (T.ones_like(recurrence_relation) - recurrence_relation)

    # alpha
    alpha_t = T.cat((pred[0, :, :2], T.ones(batch, 2*U-1).type(floatX)*eps_nan), dim=1) # (batch, 2U+1)
    probability = alpha_t[None] # (1, batch, 2U+1)
    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1, Time).type(longX):
        alpha_t = log_batch_dot(alpha_t, recurrence_relation) + pred[t]
        probability = T.cat((probability, alpha_t[None]), dim=0)

    labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    labels_prob = log_sum_exp(labels_2, labels_1)
#     pdb.set_trace()

    cost = -labels_prob
    return cost

#added SIL to nob version of rCTC
def reg_ctc_loss_nob_sil_log(pred, pred_len, token, token_len, blank=0, sil_token=-1):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)
    '''
    assert(sil_token > 0)
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps_nan = -1e8

    #token with sil:
    token_with_sil = T.cat((T.full((batch,1), sil_token).type(longX), token), dim=1) #(batch, U+1)
    token_with_sil = T.cat((token_with_sil, T.full((batch,1), blank).type(longX)), dim=1) #(batch, U+2)
    U = U + 2
    index1 = T.arange(batch)
    index2 = token_len + 1
    token_with_sil[index1, index2] = sil_token
    token = token_with_sil
    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)

    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)

    # recurrence relation, no jump from token to token is allowed. 
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1)
    # regulariztion
    reg_mat = regularized_eye(length)
    recurrence_relation = recurrence_relation * reg_mat
    recurrence_relation = eps_nan * (T.ones_like(recurrence_relation) - recurrence_relation)

    # alpha
    alpha_t = T.cat((pred[0, :, :2], T.ones(batch, 2*U-1).type(floatX)*eps_nan), dim=1) # (batch, 2U+1)
    probability = alpha_t[None] # (1, batch, 2U+1)
    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1, Time).type(longX):
        alpha_t = log_batch_dot(alpha_t, recurrence_relation) + pred[t]
        probability = T.cat((probability, alpha_t[None]), dim=0)

    labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    labels_prob = log_sum_exp(labels_2, labels_1)
#     pdb.set_trace()

    cost = -labels_prob
    return cost

def reg_ctc_cost(out, targets, sizes, target_sizes):
#    A batched version for uni_alpha_cost
#    param out: (Time, batch, voca_size+1)
#    param targets: targets without splited
#    param sizes: size for out (N)
#    param target_sizes: size for targets (N)

    Time = out.size(0)
    pred = out # ----- svail ------ input is already log_softmax (log prob)

    offset = 0
    batch = target_sizes.size(0)
    target_max = target_sizes.max().item()
    target = T.zeros(batch, target_max).type(longX)

    for index, (target_size, size) in enumerate(zip(target_sizes, sizes)):
        target[index, :target_size.item()] = targets[offset: offset+target_size.item()].data
        offset += target_size.item()

    if not cuda:
        costs = reg_ctc_loss_log(pred.cpu(), sizes.data.type(longX), target, target_sizes.data.type(longX))
    else:
        costs = reg_ctc_loss_log(pred, sizes.data.type(longX), target, target_sizes.data.type(longX))
    return costs.sum()

def reg_ctc_nob_cost(out, targets, sizes, target_sizes):
#    A batched version for uni_alpha_cost
#    param out: (Time, batch, voca_size+1)
#    param targets: targets without splited
#    param sizes: size for out (N)
#    param target_sizes: size for targets (N)

    Time = out.size(0)
    pred = out # ----- svail ------ input is already log_softmax (log prob)

    offset = 0
    batch = target_sizes.size(0)
    target_max = target_sizes.max().item()
    target = T.zeros(batch, target_max).type(longX)

    for index, (target_size, size) in enumerate(zip(target_sizes, sizes)):
        target[index, :target_size.item()] = targets[offset: offset+target_size.item()].data
        offset += target_size.item()

    if not cuda:
        costs = reg_ctc_loss_nob_log(pred.cpu(), sizes.data.type(longX), target, target_sizes.data.type(longX))
    else:
        costs = reg_ctc_loss_nob_log(pred, sizes.data.type(longX), target, target_sizes.data.type(longX))
    return costs.sum()

def reg_ctc_sil_cost(out, targets, sizes, target_sizes, sil_token):
#    A batched version for uni_alpha_cost
#    param out: (Time, batch, voca_size+1)
#    param targets: targets without splited
#    param sizes: size for out (N)
#    param target_sizes: size for targets (N)

    Time = out.size(0)
    pred = out # ----- svail ------ input is already log_softmax (log prob)

    offset = 0
    batch = target_sizes.size(0)
    target_max = target_sizes.max().item()
    target = T.zeros(batch, target_max).type(longX)

    for index, (target_size, size) in enumerate(zip(target_sizes, sizes)):
        target[index, :target_size.item()] = targets[offset: offset+target_size.item()].data
        offset += target_size.item()

    if not cuda:
        costs = reg_ctc_loss_sil_log(pred.cpu(), sizes.data.type(longX), target, target_sizes.data.type(longX), sil_token=sil_token)
    else:
        costs = reg_ctc_loss_sil_log(pred, sizes.data.type(longX), target, target_sizes.data.type(longX), sil_token=sil_token)
    return costs.sum()

def reg_ctc_nb_sil_cost(out, targets, sizes, target_sizes, sil_token):
#    A batched version for uni_alpha_cost
#    param out: (Time, batch, voca_size+1)
#    param targets: targets without splited
#    param sizes: size for out (N)
#    param target_sizes: size for targets (N)
    Time = out.size(0)
    pred = out # ----- svail ------ input is already log_softmax (log prob)

    offset = 0
    batch = target_sizes.size(0)
    target_max = target_sizes.max().item()
    target = T.zeros(batch, target_max).type(longX)

    for index, (target_size, size) in enumerate(zip(target_sizes, sizes)):
        target[index, :target_size.item()] = targets[offset: offset+target_size.item()].data
        offset += target_size.item()

    if not cuda:
        costs = reg_ctc_loss_nob_sil_log(pred.cpu(), sizes.data.type(longX), target, target_sizes.data.type(longX), sil_token=sil_token)
    else:
        costs = reg_ctc_loss_nob_sil_log(pred, sizes.data.type(longX), target, target_sizes.data.type(longX), sil_token=sil_token)
    return costs.sum()
    
def reg_ctc_nob_mid_cost(out, targets, sizes, target_sizes):
#    A batched version for uni_alpha_cost
#    param out: (Time, batch, voca_size+1)
#    param targets: targets without splited
#    param sizes: size for out (N)
#    param target_sizes: size for targets (N)

    Time = out.size(0)
    pred = out # ----- svail ------ input is already log_softmax (log prob)

    offset = 0
    batch = target_sizes.size(0)
    target_max = target_sizes.max().item()
    target = T.zeros(batch, target_max).type(longX)

    for index, (target_size, size) in enumerate(zip(target_sizes, sizes)):
        target[index, :target_size.item()] = targets[offset: offset+target_size.item()].data
        offset += target_size.item()

    if not cuda:
        costs = reg_ctc_loss_nob_mid_log(pred.cpu(), sizes.data.type(longX), target, target_sizes.data.type(longX))
    else:
        costs = reg_ctc_loss_nob_mid_log(pred, sizes.data.type(longX), target, target_sizes.data.type(longX))
    return costs.sum()