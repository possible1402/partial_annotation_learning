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

import argparse
from ast import arg
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import pad
from torch.utils.data import DataLoader, RandomSampler,Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import re
import pprint
from transformers import AutoTokenizer, AutoModelForTokenClassification
import math
sys.path.append(r'/Users/ding/Desktop/snellius/NER/TS-BERT-Partial-CRF_release')
os.chdir(sys.path[-1])
print(sys.path[-1])
from models.transformers import WEIGHTS_NAME,BertForTokenClassification,BertConfig,BertTokenizer,PubmedBERT_Partial_CRF
from models.transformers import RobertaConfig,RobertaForTokenClassification,RobertaTokenizer
from data_utils_self import load_and_cache_examples, get_labels,read_examples_from_file,load_and_cache_one_example

import shutil

import pandas as pd
import pickle
pattern = re.compile(r'\d+\n')
import logging
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import (
    AdamW,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from model_utils import multi_source_label_refine_for_each_category
import math


logger = logging.getLogger(__name__)



def set_seed(conf):
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)


def initialize(conf, model, t_total, epoch):

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": conf.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=conf.learning_rate, \
                eps=conf.adam_epsilon, betas=(conf.adam_beta1,conf.adam_beta2))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=conf.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if epoch == 0:
            if os.path.isfile(os.path.join(conf.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(conf.model_name_or_path, "scheduler.pt")
            ):
                # Load in optimizer and scheduler states
                optimizer.load_state_dict(torch.load(os.path.join(conf.model_name_or_path, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(conf.model_name_or_path, "scheduler.pt")))

        if conf.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=conf.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if conf.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if conf.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[conf.local_rank], output_device=conf.local_rank, find_unused_parameters=True
            )

        model.zero_grad()
        return model, optimizer, scheduler

def evaluate(args, model, tokenizer, labels,pad_token_label_id, mode, global_step,prefix="", verbose=True,train_span_set=None):
    eval_dataset,_ = load_and_cache_examples(args, tokenizer, labels,pad_token_label_id, mode=mode,train_span_set=train_span_set)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation %s *****", prefix)
    if verbose:
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    golden_label_list = []
    preds_list = []
    model.eval()
    label_map_reverse={value:key for key,value in args.label_map.items()}
    label_map_reverse={key.replace('L-','E-').replace('U-','S-'):value for key,value in label_map_reverse.items()}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[5],"original_ids":batch[3],"entity_removal_tag":batch[9],"label_mask":batch[10]} # attention_mask: [batch size, max seq len]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss,pred_tags,golden_tags,logits  = outputs['loss'],outputs['pred_tags'],outputs['tags'],outputs['tag_prob']
            golden_label_list.extend(golden_tags)  # 已经去除了mask
            preds_list.extend(pred_tags)
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    results = {
        f"{mode}_eval_loss": eval_loss,
        # default micro
        f"{mode}_precision": precision_score(golden_label_list, preds_list,scheme='IOBES'),
        f"{mode}_recall": recall_score(golden_label_list, preds_list,scheme='IOBES'),
        f"{mode}_f1": f1_score(golden_label_list, preds_list,scheme='IOBES')
    }
    print(results)

    
    return results,eval_loss


def get_classwise_confidence(all_result,global_step):
    label_confidence_map={kk+1:1.0 for kk in range(len(label_map)-1)} # 其实应该初始化为1，否则为原标签，而不是设置为0，然后
    pred_labels=[f'score_class_{str(kk+1)}' for kk in range(len(label_map.keys())-1)]
    pred_id=all_result[pred_labels].idxmax(axis=1).str.replace('score_class_','').map(int)
    # pred_score=all_result[pred_labels].max(axis=1)
    all_result['Correct']=(pred_id==all_result['golden_labels'].map(int))
    all_result['pred_ids']=pd.DataFrame(pred_id)
    # 所有正确位置，各个标签的confidence取平均
    for label in pred_labels:
        label_confidence_map[int(label.split('_')[-1])]=all_result[(all_result['Correct']==True)&(all_result['golden_labels'].map(str)==label.split('_')[-1])][label].mean()
    label_confidence_map[1]=all_result[(pred_id==1)]['score_class_1'].mean() # O标签的阈值是所有预测标签为O的confidence取平均
    for key,value in label_confidence_map.items(): # 将其中的nan替换为1.0
        if np.isnan(value):
            label_confidence_map[key]=1.0
    label_confidence_map_frame=pd.Series(label_confidence_map).to_frame().transpose()
    global_step_frame=pd.DataFrame([global_step])
    removal_rate_frame=pd.DataFrame([conf.entity_removal_rate])
    removal_algorithm=pd.DataFrame([conf.entity_removal_method])
    entity_name_frame=pd.DataFrame([conf.entity_name])
    confidence_map_frame=pd.concat([global_step_frame,removal_rate_frame,removal_algorithm,entity_name_frame,label_confidence_map_frame],axis=1,ignore_index=True)
    confidence_map_frame.columns=['global_step','removal_rate','removal_algorithm','entity_name']+pred_labels
    return label_confidence_map



def self_training(batch,label_confidence_map,global_step,self_training_teacher_model):

    inputs = {"input_ids": batch[0], "attention_mask": batch[1],"labels": batch[5],"original_ids":batch[3],"entity_removal_tag":batch[9]} 
    if conf.model_type != "distilbert":
        inputs["token_type_ids"] = (batch[2] if conf.model_type in ["bert", "xlnet"] else None)
    with torch.no_grad():
        outputs = self_training_teacher_model(**inputs)# # (loss), logits,sequence_output,(hidden_states), (attentions) 
    
    pred_labels,pseudo_label_mask=multi_source_label_refine_for_each_category(conf,outputs,global_step,label_confidence_map)# label_mask:[batch size,max seq len],pred_labels[batch size,max seq len, num labels]
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": pred_labels,"original_ids":batch[3],"entity_removal_tag":batch[9],"label_mask":batch[10],"pseudo_label_mask":pseudo_label_mask}
    return  inputs 

def get_pred_result(outputs,global_step,mode):
    result={}
    label_map_reverse={value:key for key,value in conf.label_map.items()}
    label_map_reverse={key.replace('L-','E-').replace('U-','S-'):value for key,value in label_map_reverse.items()}
    golden_tags,logits,mask,original_labels,entity_removal_tag  = outputs['tags'],outputs['tag_prob'],outputs['attention_mask'],outputs['original_ids'],outputs['entity_removal_tag']
    non_masked_logits=logits.masked_select(mask.unsqueeze(2)).view(-1,logits.shape[-1])[:,1:].cpu().detach().numpy() # non_masked_logits:[number of non-mask tokens in the batch, num classes],去除mask的得分，以及没有latent tag的
    non_mask_golden_tags=[ele for kk in golden_tags for ele in kk] # 这里面有latent_tag
    assert non_masked_logits.shape[0]==len(non_mask_golden_tags)
    result['global_step']=pd.DataFrame([global_step]*len(non_mask_golden_tags))
    result['removal_rate']=pd.DataFrame([conf.entity_removal_rate]*len(non_mask_golden_tags))
    result['seed']=pd.DataFrame([conf.seed]*len(non_mask_golden_tags))
    result['removal_algorithm']=pd.DataFrame([conf.entity_removal_method]*len(non_mask_golden_tags))
    result['entity_name']=pd.DataFrame([conf.entity_name]*len(non_mask_golden_tags))
    result['token_id']=pd.DataFrame(range(0,non_masked_logits.shape[0]))
    result['golden_labels']=pd.DataFrame([label_map_reverse[kk] for kk in non_mask_golden_tags])
    result['original_labels']=pd.DataFrame(original_labels.masked_select(mask).contiguous().view(-1).cpu().detach().numpy())
    result['entity_removal_tag']=pd.DataFrame(entity_removal_tag.masked_select(mask).contiguous().view(-1).cpu().detach().numpy())
    for kk in range(0,non_masked_logits.shape[1]): # num classes
        result[f'score_class_{str(kk+1)}']=pd.DataFrame(non_masked_logits[:,kk]) # 0是latent tag，但是预测标签不可能有latent tag
    final_result=pd.concat(result.values(),axis=1,ignore_index=True)
    final_result.columns=result.keys()
    return final_result

def best_model(model,tokenizer,dev_results,best_f1,best_precision,best_recall,global_step,train_span_set,stop_flag):
    # save the best model
    f1 = dev_results["dev_f1"]
    if f1 > best_f1: 
        print("~~~~~~~NEW BEST Model~~~~~~~~~~~")
        best_f1=f1
        best_precision=dev_results['dev_precision']
        best_recall=dev_results['dev_recall']
        logger.info(" best_f1 = %s", best_f1)
        logger.info("***** Eval results *****")
        output_dir = os.path.join(
            conf.output_dir,  "checkpoint-best")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model,
                                                "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(conf, os.path.join(
            output_dir, "training_conf.bin"))
        logger.info(
            "Saving model checkpoint to %s", output_dir)
        _, _  = evaluate(conf, model, tokenizer, labels,pad_token_label_id, mode="test", global_step=global_step, verbose=False,train_span_set=train_span_set)
    else:
        stop_flag+=1 # early stopping

    model.to(conf.device)
    return best_f1,best_precision,best_recall,stop_flag



def train(conf, train_dataset, model, tokenizer, labels, pad_token_label_id,train_span_set):
    """ Train the model """
    best_f1 = 0
    best_precision=0
    best_recall=0
    stop_flag=0
    all_train_result=pd.DataFrame()

    label_confidence_map={kk+1:0.0 for kk in range(len(label_map))} # 不包含latent，其实O的置信度应该也是缺失的，因为原始标签中没有O，所以不可能预测标签等于原始标签
    conf.train_batch_size = conf.per_gpu_train_batch_size * max(1, conf.n_gpu)
    train_sampler = RandomSampler(train_dataset) if conf.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=conf.train_batch_size)
    t_total = len(train_dataloader) // conf.gradient_accumulation_steps * conf.num_train_epochs

    model, optimizer, scheduler = initialize(conf, model, t_total, 0)
    # conf.logging_steps=math.ceil(len(train_dataset)/conf.per_gpu_train_batch_size) # 每个epoch记录一次
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", conf.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", conf.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        conf.train_batch_size
        * conf.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if conf.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", conf.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(conf.num_train_epochs), desc="Epoch", disable=conf.local_rank not in [-1, 0]
    )
    set_seed(conf)  # Added here for reproductibility
    self_training_teacher_model = model
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=conf.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator): # batch:[token_id,attention_mask,token_type_ids,original_label,full_label,latent_label,guid-1,input_len,guid]
            # print(f'正在进行第{step}次iteration')
            model.train()
            batch = tuple(t.to(conf.device) for t in batch)

            # Update labels periodically after certain begin step
            if global_step >= conf.self_training_begin_step:
                delta = global_step - conf.self_training_begin_step
                if delta % conf.self_training_period == 0: # 在每次self_training_period时更新模型
                    self_training_teacher_model = copy.deepcopy(model)
                inputs=self_training(batch,label_confidence_map,global_step,self_training_teacher_model)
                
            else:  # torch.argmax(pred_labels[0,:,:],dim=1)进行检验;      
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[5],"original_ids":batch[3],"entity_removal_tag":batch[9],"label_mask":batch[10]} # original_id是进行entity去除之前的标签

            if conf.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if conf.model_type in ["bert", "xlnet"] else None
                )
            # inputs中包含inputs_ids: [batch size, max seq len],inputs['attention_mask']:[batch size, max seq len],labels:[batch size, max seq len],token)_type_ids:[batch size, max seq len]
            # 在self_training_begin_step之后，学生模型采用含label_mask的输入进行训练 # 使用logits[0,1,:]，torch.max(logits[0,1,:]),torch.argmax(logits[0,1,:])进行调试
            outputs = model(**inputs) # # outputs是一个字典，包含local_potentials, pred_crf, pred tags, constrained_pred_crf, loss, tags,metrics
            loss = outputs['loss']

            # logits:[batch size,max seq len,num labels]


            mt_loss, vat_loss = 0, 0 # logits:[batch size, max seq len, num labels];final_emebeds:[batch size, max seq len,768]
            loss = loss + conf.mt_beta * mt_loss + conf.vat_beta * vat_loss
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % conf.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if conf.local_rank in [-1, 0] and conf.logging_steps > 0 and global_step % conf.logging_steps == 0:
                    # Log metrics
                    if conf.evaluate_during_training:
                        
                        logger.info("***** Entropy loss: %.4f, mean teacher loss : %.4f; vat loss: %.4f *****", \
                            loss - conf.mt_beta * mt_loss - conf.vat_beta * vat_loss, \
                            conf.mt_beta * mt_loss, conf.vat_beta * vat_loss)
                        dev_results, _ = evaluate(conf, model, tokenizer, labels,pad_token_label_id, mode="dev",global_step=global_step, prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, conf.num_train_epochs), verbose=False,train_span_set=train_span_set)
                        best_f1,best_precision,best_recall,stop_flag=best_model(model,tokenizer,dev_results,best_f1,best_precision,best_recall,global_step,train_span_set=train_span_set,stop_flag=stop_flag)
         
                        if stop_flag==7: 
                            if best_f1==0:
                                _, _  = evaluate(conf, model, tokenizer, labels,pad_token_label_id, mode="test", global_step=global_step,prefix='', verbose=False,train_span_set=train_span_set)
                            return model, global_step, tr_loss / global_step


                if global_step <=conf.self_training_begin_step and global_step>=conf.warmup_steps: # 继续追踪自训练之后在训练集上相关指标

                    train_result=get_pred_result(outputs,global_step,mode='train') # 当前step的warmup
                    if global_step <= conf.self_training_begin_step : # 计算confidence threshold
                        all_train_result=pd.concat([all_train_result,train_result],axis=0)
                    if global_step == conf.self_training_begin_step:
                        label_confidence_map=get_classwise_confidence(all_train_result,global_step)
                        print(f'confidence threshold is: {label_confidence_map}')
                log_metrics={}
                for key,value in outputs['metrics'].items():
                    log_metrics['training/'+key]=value


    if best_f1==0:
        _, _  = evaluate(conf, model, tokenizer, labels,pad_token_label_id, mode="test", global_step=global_step,prefix='', verbose=False,train_span_set=train_span_set)

    return model, global_step, tr_loss / global_step




def parse_argument(parser):

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for theNER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: "
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list:,"
    )
    parser.add_argument(
        "--data_cahce_index",
        default='bert-base-chinese',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="BETA1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="BETA2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--labels", default="", type=str,help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--label_index", type=int, default=1,help="the column index of label")
    parser.add_argument('--visible_device', default="0")
    parser.add_argument("--train_file", type=str, default="train.tsv", help="the file to train")
    parser.add_argument("--dev_file", type=str, default="dev.tsv", help="the file to evaluate")
    parser.add_argument("--test_file", type=str, default="test.tsv", help="the file to predict")
    parser.add_argument("--write_examples_to_file", action="store_true",help="Whether to load one example each time")
    parser.add_argument('--evaluation_method', default="seqeval", type=str, help="evaluation method, choices = [set_eval, seqeval]")
    parser.add_argument("--shuffle_data", action="store_true",help="Whether shuffle data")

    # self-training
    parser.add_argument('--whether_self_training',action="store_true", help = 'whether to use teacher-student self training')
    parser.add_argument('--self_training_reinit', type = int, default = 0, help = 're-initialize the student model if the teacher model is updated.')
    parser.add_argument('--self_training_begin_step', type = int, default = 900, help = 'the begin step (usually after the first epoch) to start self-training.')
    parser.add_argument('--self_training_label_mode', type = str, default = "hard", help = 'pseudo label type. choices:[hard(default), soft].')
    parser.add_argument('--self_training_period', type = int, default = 878, help = 'the self-training period.')
    parser.add_argument('--self_training_hp_label', type = float, default = -1, help = 'use high precision label.')
    parser.add_argument('--self_training_hp_label_category', type = int, default = -1, help = 'for category oriented self training ,decide whether to add false labels to loss calculation,choices:[0(not included),1(included)]')
    parser.add_argument('--self_training_ensemble_label', type = int, default = 0, help = 'use ensemble label.')
    parser.add_argument("--whether_category_oriented", action="store_true",help="whether to use different category distribution for psdudo label")
    parser.add_argument("--confidence_test", action="store_true",help="whether to use confidence test code")
    parser.add_argument("--update_every_period", action="store_true",help="whether update confidence every self training period")
    parser.add_argument("--update_scheme", type=str,choices=['update_all','update_entity'],default='update_all')


    # entity removal
    parser.add_argument("--entity_removal_method", type=str,choices=['remove_annotations_randomly','remove_surfaceforms_randomly','remove_annotations_randomly_add_positive','remove_surfaceforms_randomly_add_positive','None'],default='remove_annotations_randomly')
    parser.add_argument("--entity_removal_rate", type=float,default=0.0)
    parser.add_argument("--do_remove_entity", action="store_true")

    # entity ratio
    parser.add_argument("--self_training_loss_weight", type=float,default=5.0)
    parser.add_argument("--prior_loss_weight", type=float,default=10.0)
    parser.add_argument("--entity_ratio", type=float,default=0.01)
    parser.add_argument("--entity_ratio_margin", type=float,default=0.05)
    parser.add_argument("--overall_eer_weight", type=float,default=10.0)
    conf = parser.parse_args()

    return conf

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    conf=parse_argument(parser)
    conf.entity_name=conf.data_dir.split('/')[-1]
    print("The path of the dataest: {}".format(conf.data_dir))
    conf.output_dir=os.path.join(conf.output_dir,os.path.split(conf.data_dir)[-1],conf.entity_removal_method,str(conf.entity_removal_rate),f'begin{conf.self_training_begin_step}_period{conf.self_training_period}_epoch{conf.num_train_epochs}_LR{conf.learning_rate}_seed{conf.seed}')
    conf.vocab_dir=os.path.join(conf.data_dir,'pubmedbert-entity.vocab')

    if (
        os.path.exists(conf.output_dir)
        and os.listdir(conf.output_dir)
        and conf.do_train
        and not conf.overwrite_output_dir
    ):
        print('delete output dir')
        shutil.rmtree(conf.output_dir)

    # Create output directory if needed
    if not os.path.exists(conf.output_dir) and conf.local_rank in [-1, 0]:
        os.makedirs(conf.output_dir)

    device = torch.device(f"cuda:{conf.visible_device}")  # select the device number you want to use

    # Set the device for PyTorch tensors
    torch.cuda.set_device(device)
    conf.device = device
    conf.n_gpu = 1
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if conf.local_rank in [-1, 0] else logging.WARN,
    )
    logging_fh = logging.FileHandler(os.path.join(conf.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        conf.local_rank,
        device,
        conf.n_gpu,
        bool(conf.local_rank != -1),
        conf.fp16,
    )

    # Set seed
    set_seed(conf)
    labels = get_labels(conf.labels)
    labels.insert(0,'_') # add latent label
    num_labels = len(labels)

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index
    label_map={kk:labels[kk] for kk in range(len(labels))}
    conf.label_map=label_map
    # Load pretrained model and tokenizer
    if conf.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if conf.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", conf)
        
    model_config = BertConfig.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                        num_labels=num_labels,
                                        cache_dir=conf.cache_dir if conf.cache_dir else None)       
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                                do_lower_case=True,
                                                cache_dir=conf.cache_dir if conf.cache_dir else None)       
    model = PubmedBERT_Partial_CRF.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                        from_tf=bool(".ckpt" in conf.model_name_or_path),
                                        config=model_config,labels=label_map,vocab_dir=conf.vocab_dir,
                                        cache_dir=conf.cache_dir if conf.cache_dir else None,self_training_loss_weight=conf.self_training_loss_weight,prior_loss_weight=conf.prior_loss_weight,entity_ratio=conf.entity_ratio,entity_ratio_margin=conf.entity_ratio_margin,overall_eer_weight=conf.overall_eer_weight)   
    model.to(conf.device)
    set_seed(conf)

    train_dataset,train_span_set = load_and_cache_examples(conf, tokenizer, labels,pad_token_label_id, mode="train")

    model.to(conf.device)
    model, global_step, tr_loss=train(conf, train_dataset, model, tokenizer, labels,pad_token_label_id,train_span_set)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


