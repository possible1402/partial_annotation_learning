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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import re
from entity_removal import remove_annotations_randomly,remove_surfaceforms_randomly,get_all_entity_spans,add_flag

logger = logging.getLogger(__name__)
pattern=re.compile(r'\d+\n')
import copy
class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words,hp_labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.hp_labels = hp_labels
        self.original_labels=copy.deepcopy(hp_labels)
        self.flag = [0]*len(hp_labels) # indicate if this token is a removed entity token


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, original_label_ids,full_label_ids,hp_label_ids,input_len,guid,tokens,removal_entity_flag):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.original_label_ids = original_label_ids
        self.full_label_ids = full_label_ids
        self.hp_label_ids = hp_label_ids
        self.input_len=input_len
        self.guid=guid
        self.tokens=tokens
        self.removal_entity_flag=removal_entity_flag





def read_examples_from_file(args, mode,label_index=1,whether_write_to_file=False):

    if mode=='train':
        file_path=os.path.join(args.data_dir, args.train_file)
    elif mode=='dev':
        file_path=os.path.join(args.data_dir, args.dev_file)
    elif mode=='test':
        file_path=os.path.join(args.data_dir, args.test_file)
        
    example_file_path=os.path.join(args.data_dir, "{}_examples.tsv".format(mode))
    guid_index = 1
    examples = []
    if os.path.exists(example_file_path) and whether_write_to_file==True:
        raise ValueError("example_file exists")

    with open(file_path, encoding="utf-8") as f:
        words = []
        hp_labels=[]
        for line in f: # 相当于f.readline()
            if pattern.match(line) or line == "" or line == "\n" or line=='-DOCSTART - O\n':
            # if line.startswith("-DOCSTART -") or line == "" or line == "\n":
                if words:
                    example=InputExample(guid=guid_index,
                                                words=words,
                                                hp_labels=hp_labels)
                    examples.append(example)
                    guid_index += 1
                    words = []
                    hp_labels=[]
                    if whether_write_to_file:
                        with open(example_file_path, 'a', encoding='utf-8') as z:
                            z.write(example.guid + '\t' + str(example.words) + '\t' + str(example.labels) +'\n')

            else:
                splits = line.split()
                words.append(splits[0].strip())
                #
                cur_label=splits[label_index].replace("\n", "")
                hp_labels.append(cur_label)
                  
        if words:
            example=InputExample(guid=guid_index,
                                        words=words,
                                        hp_labels=hp_labels)
            examples.append(example)
            if whether_write_to_file:
                with open(example_file_path, 'a', encoding='utf-8') as z:
                    z.write(example.guid + '\t' + str(example.words) + '\t' + str(example.hp_labels) +'\n')
    return examples


def generate_BIOUL_label_sequence(mode,hp_label,hp_label_ids,label_map,word_tokens):
    if hp_label=='O':
        if mode=='train':
            hp_label_ids.extend([label_map['_']]*(len(word_tokens)))
        else:
            hp_label_ids.extend([label_map['O']]*(len(word_tokens)))
    else:
        type=hp_label.split('-')[-1]
        if hp_label.startswith('B-'):
            hp_label_ids.extend([label_map[hp_label]] + [label_map[f'I-{type}']] * (len(word_tokens) - 1))
        elif hp_label.startswith('I-'):
            hp_label_ids.extend([label_map[hp_label]] * (len(word_tokens)))
        elif hp_label.startswith('L-'):
            hp_label_ids.extend([label_map[f'I-{type}']] * (len(word_tokens)-1)+[label_map[hp_label]])
        elif hp_label.startswith('U-'):
            if len(word_tokens)>2:
                hp_label_ids.extend([label_map[f'B-{type}']] +[label_map[f'I-{type}']]* (len(word_tokens)-2)+[label_map[f'L-{type}']])
            elif len(word_tokens)==1:
                hp_label_ids.extend([label_map[f'U-{type}']])
            elif len(word_tokens)==2:
                hp_label_ids.extend([label_map[f'B-{type}']] +[label_map[f'L-{type}']])        
    return hp_label_ids

def convert_examples_to_features(args,mode,
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    show_exnum = 5,
    train_span_set=None
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    extra_long_samples = 0

    if args.entity_removal_method!='None' and mode=='train': # entity removal
        if args.entity_removal_method=='remove_annotations_randomly':
            number_of_annotations,span_set,removal_annotations=remove_annotations_randomly(args,examples)
        elif args.entity_removal_method=='remove_surfaceforms_randomly':
            number_of_annotations,span_set,removal_annotations=remove_surfaceforms_randomly(args,examples)
        elif args.entity_removal_method=='remove_annotations_randomly_add_positive': 
            number_of_annotations,span_set,removal_annotations=remove_annotations_randomly(args,examples,True)
        elif args.entity_removal_method=='remove_surfaceforms_randomly_add_positive':
            number_of_annotations,span_set,removal_annotations=remove_surfaceforms_randomly(args,examples,True)
        else:
            raise

    else:
        all_spans=get_all_entity_spans(examples,"hp_labels")
        number_of_annotations=len(all_spans)
        span_set=[]
        removal_annotations=[]
    if mode!='train':
        add_flag(train_span_set,examples)

    for (ex_index, example) in enumerate(examples): 
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        full_label_ids = [] 
        hp_label_ids = [] 
        origianl_label_ids=[]
        removal_entity_flag=[]
        for word, hp_label,original_label,flag in zip(example.words,example.hp_labels,example.original_labels,example.flag): # 这里的label和original_label都是一样的，并没有经过改变
            word_tokens = tokenizer.tokenize(word)
            if(len(word_tokens) == 0):
                continue
            tokens.extend(word_tokens)
            
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            full_label_ids.extend([label_map[hp_label]] * len(word_tokens))
            removal_entity_flag.extend([flag]* len(word_tokens))
            hp_label_ids=generate_BIOUL_label_sequence(mode,hp_label,hp_label_ids,label_map,word_tokens)
            origianl_label_ids=generate_BIOUL_label_sequence(mode,original_label,origianl_label_ids,label_map,word_tokens)
        assert len(tokens)==len(hp_label_ids)==len(full_label_ids)==len(origianl_label_ids)
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            hp_label_ids = hp_label_ids[: (max_seq_length - special_tokens_count)]
            full_label_ids = full_label_ids[: (max_seq_length - special_tokens_count)]
            origianl_label_ids = origianl_label_ids[: (max_seq_length - special_tokens_count)]
            removal_entity_flag = removal_entity_flag[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1


        tokens += [sep_token]
        hp_label_ids += [label_map['O']]
        full_label_ids += [pad_token_label_id]
        origianl_label_ids += [label_map['O']]
        removal_entity_flag += [0]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            hp_label_ids += [label_map['O']]
            full_label_ids += [pad_token_label_id]
            removal_entity_flag += [0]
            origianl_label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            hp_label_ids += [label_map['O']]
            origianl_label_ids += [label_map['O']]
            full_label_ids += [pad_token_label_id]
            removal_entity_flag += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            hp_label_ids = [label_map['O']] + hp_label_ids
            origianl_label_ids = [label_map['O']] + origianl_label_ids
            full_label_ids = [pad_token_label_id] + full_label_ids
            removal_entity_flag = [0]+removal_entity_flag
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_len=len(hp_label_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            if mode=='train':
                hp_label_ids = ([label_map['_']] * padding_length) + hp_label_ids
                origianl_label_ids = ([label_map['_']] * padding_length) + origianl_label_ids
            else:
                hp_label_ids=([label_map['O']] * padding_length) + hp_label_ids
                origianl_label_ids=([label_map['O']] * padding_length) + origianl_label_ids
            full_label_ids = ([pad_token_label_id] * padding_length) + full_label_ids
            removal_entity_flag=([0] * padding_length) + removal_entity_flag
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            if mode=='train':
                hp_label_ids += [label_map['_']] * padding_length
                origianl_label_ids += [label_map['_']] * padding_length
            else:
                hp_label_ids += [label_map['O']] * padding_length
                origianl_label_ids += [label_map['O']] * padding_length
            full_label_ids += [pad_token_label_id] * padding_length
            removal_entity_flag+=[0] * padding_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(hp_label_ids) == max_seq_length
        assert len(origianl_label_ids) == max_seq_length
        assert len(full_label_ids) == max_seq_length
        assert len(removal_entity_flag) == max_seq_length

        if ex_index < show_exnum:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("hp_label_ids: %s", " ".join([str(x) for x in hp_label_ids]))
            logger.info("origianl_label_ids: %s", " ".join([str(x) for x in origianl_label_ids]))
            logger.info("full_label_ids: %s", " ".join([str(x) for x in full_label_ids]))
            logger.info("input_len: %d", input_len)

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, original_label_ids=origianl_label_ids, full_label_ids=full_label_ids, hp_label_ids=hp_label_ids,input_len=input_len,guid=int(example.guid),tokens=tokens,removal_entity_flag=removal_entity_flag)
        )
    logger.info("Extra long example %d of %d", extra_long_samples, len(examples))
    return features,number_of_annotations,span_set,removal_annotations


def load_and_cache_examples(args, tokenizer, labels,pad_token_label_id, mode,train_span_set=None):
    if mode!='train':
        cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}_{}".format(mode,
                                                                                    list(filter(None,
                                                                                                args.data_cahce_index.split(
                                                                                                    "/"))).pop(),
                                                                                    str(args.max_seq_length),args.entity_removal_rate,args.entity_removal_method)) # 为每一个removal rate和removal algorithm的测试集和评估集保存文件，这样就可以避免在串行运行时读取文件报错
    else:
        cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}_{}_seed{}".format(mode,
                                                                                    list(filter(None,
                                                                                                args.data_cahce_index.split(
                                                                                                    "/"))).pop(),
                                                                                    str(args.max_seq_length),args.entity_removal_rate,args.entity_removal_method,args.seed)) # 为每一个removal rate和removal algorithm的测试集和评估集保存文件，这样就可以避免在串行运行时读取文件报错
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args, mode,label_index=args.label_index,whether_write_to_file=args.write_examples_to_file)
        features,number_of_annotations,span_set,removal_annotations = convert_examples_to_features(args,mode,
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,train_span_set=train_span_set
        )
        if args.local_rank in [-1, 0]: 
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        if mode=='train':
            print(f'number_of_annotations:{number_of_annotations}')
            print(f'number_of_removal_surfaceforms:{len(span_set)}')
            print(f'number_of_removal_annotations:{len(removal_annotations)}')
            train_span_set=span_set
        else:
            train_span_set=None

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_original_label_ids = torch.tensor([f.original_label_ids for f in features], dtype=torch.long)
    all_full_label_ids = torch.tensor([f.full_label_ids for f in features], dtype=torch.long)
    all_hp_label_ids = torch.tensor([f.hp_label_ids for f in features], dtype=torch.long)
    all_input_len=torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_guid=torch.tensor([f.guid for f in features], dtype=torch.long)
    all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long) # [0,len(features)-1]
    all_label_mask=(all_hp_label_ids!=0)&(all_hp_label_ids!=1) 
    all_entity_removal_tags=torch.tensor([f.removal_entity_flag for f in features], dtype=torch.int)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_original_label_ids, all_full_label_ids, all_hp_label_ids, all_ids,all_input_len,all_guid,all_entity_removal_tags,all_label_mask) # 其实all_full_label_ids,all_hp_label_ids,all_ids,all_input_len都没有使用
    return dataset,train_span_set


def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O","B","I"]

   

def tag_to_id(path = None):
    if path and os.path.exists(path + "tag_to_id.json"):
        with open(path + "tag_to_id.json", 'r') as f:
            data = json.load(f)
        return data
    else:
        return {"O": 0, "B-LOC": 1, "B-ORG": 2, "B-PER": 3, "B-MISC": 4, "I-PER": 5, "I-MISC": 6, "I-ORG": 7, "I-LOC": 8}


if __name__ == '__main__':
    pass
