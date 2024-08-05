# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import json
import torch.nn.functional as F
import torch
import math
import pandas as pd

logger = logging.getLogger(__name__)

    

def multi_source_label_refine_for_each_category(conf,outputs,global_step,label_confidence_map):

    result={}
    pseudo_label_mask=torch.zeros_like(outputs['attention_mask'])
    label_map_reverse={value:key for key,value in conf.label_map.items()}
    label_map_reverse={key.replace('L-','E-').replace('U-','S-'):value for key,value in label_map_reverse.items()}
    golden_ids,golden_tags,logits,mask,original_labels  = outputs['golden_ids'],outputs['tags'],outputs['tag_prob'],outputs['attention_mask'],outputs['original_ids']
    non_masked_logits=logits.masked_select(mask.unsqueeze(2)).view(-1,logits.shape[-1])[:,1:].cpu().detach()# non_masked_logits:[number of non-mask tokens in the batch, num classes],去除mask的得分，以及没有latent tag的
    non_mask_golden_tags=[ele for kk in golden_tags for ele in kk] 
    assert non_masked_logits.shape[0]==len(non_mask_golden_tags)

    max_prediction=logits[:,:,1:].max(dim=-1)
    _confidence = max_prediction[0] 
    pred_labels= max_prediction[1]+1 # pred_labels:[batch size,num of tokens]pred_labels.apply_(lambda x:label_confidence_map[x])
    _threshold=pred_labels.to(torch.float).cpu().detach().clone().apply_(lambda x:label_confidence_map[x]).to(conf.device)# [label_confidence_map[yy] for kk in pred_labels.tolist() for yy in kk]
    non_mask_pred_labels_before=non_masked_logits.max(dim=-1)[1]+1 # [batch size*num of tokens]
    if conf.update_scheme=='update_entity':
        pred_labels[(_confidence< _threshold)]=golden_ids[(_confidence< _threshold)]
        pred_labels[golden_ids==0]=0 
    elif conf.update_scheme=='update_all':
        pred_labels[_confidence< _threshold] = golden_ids[_confidence< _threshold]
        pseudo_label_mask[(_confidence>= _threshold)|(golden_ids!=0)]=1 
    pred_labels[golden_ids==1]=1  

    return pred_labels,pseudo_label_mask  # pred_labels[batch size,num tokens]

