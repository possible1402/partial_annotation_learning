
import random
import numpy as np

class Span:

    def __init__(self, left: int, right: int, type: str, inst_id: int = None):
        self.left = left
        self.right = right
        self.type = type
        self.inst_id = inst_id

class Token:

    def __init__(self, label: str, inst_id: int, token_id: int):
        self.label = label
        self.inst_id = inst_id
        self.token_id = token_id


def add_flag(span_set,examples):
    span_set=[kk.split('\t')[-1] for kk in span_set]
    all_spans=get_all_entity_spans(examples,"original_labels")
    for span_str in span_set:
        for cur_span in all_spans:
            cur_span_str = ' '.join(examples[cur_span.inst_id].words[cur_span.left:(cur_span.right + 1)]) #如果这个span的surfaceform和目标span的surfaceform相同，则对其进行修改
            if cur_span_str==span_str:
                for j in range(cur_span.left, cur_span.right + 1):
                    examples[cur_span.inst_id].flag[j]=1


def get_all_entity_spans(examples,ex_label):
    
    all_spans = []
    for example in examples:
        labels = example.__dict__[ex_label]
        start = -1
        for i in range(len(labels)):
            if labels[i].startswith("B-"):
                start = i
            if labels[i].startswith("L-"):
                end = i
                all_spans.append(Span(start, end, labels[i][2:], inst_id=example.guid-1))
            if labels[i].startswith("U-"):
                all_spans.append(Span(i, i, labels[i][2:], inst_id=example.guid-1)) # 因为guid从1开始
    return all_spans

def remove_annotations_randomly(config,examples,add_false_positives=False):
    """
    Remove certain number of entities and make them become O label
    :param examples:
    :param config:
    :return:
    """
    all_spans=get_all_entity_spans(examples,"hp_labels")
    random.shuffle(all_spans)
    removal_annotations=[]
    span_set = set()
    number_of_annotations=len(all_spans)
    num_entity_removed = round(number_of_annotations * config.entity_removal_rate)
    for i in range(num_entity_removed):
        span = all_spans[i]
        id = span.inst_id 
        output = examples[id].hp_labels
        for j in range(span.left, span.right + 1):
            output[j] = 'O'  
        span_str = ' '.join(examples[id].words[span.left:(span.right + 1)])
        span_str = span.type + "\t" + span_str
        span_set.add(span_str)
        removal_annotations.append(span_str)
    new_all_spans=get_all_entity_spans(examples,"hp_labels")
    if add_false_positives:
        num_of_entity_tokens=np.sum(list(map(lambda span:span.right+1-span.left,new_all_spans)))
        num_false_positives=round(num_of_entity_tokens*config.false_positive_rate)
        add_false_positives_to_examples(config,examples,num_false_positives,new_all_spans) 
    assert len(removal_annotations)==num_entity_removed
    assert len(new_all_spans)==len(all_spans)-num_entity_removed
    return number_of_annotations,span_set,removal_annotations


def remove_surfaceforms_randomly(config,examples,add_false_positives=False):
    # ori_span_str=[span.type + "\t" +' '.join(train_insts[kk.inst_id].input.words[kk.left:(kk.right + 1)]) for kk in all_spans]
    all_spans=get_all_entity_spans(examples,"hp_labels")
    number_of_annotations=len(all_spans)
    random.shuffle(all_spans) 
    flag=0
    num_entity_removed = round(number_of_annotations * config.entity_removal_rate)
    
    span_set = set()
    removal_annotations=[]
    for i in range(len(all_spans)):
        if flag<round(num_entity_removed*0.99):
            span = all_spans[i] 
            id = span.inst_id
            span_str = ' '.join(examples[id].words[span.left:(span.right + 1)])
            span_str_with_type = span.type + "\t" + span_str
            if span_str_with_type not in span_set: 
                span_set.add(span_str_with_type)
                for cur_span in all_spans:
                    cur_span_str = ' '.join(examples[cur_span.inst_id].words[cur_span.left:(cur_span.right + 1)])
                    if cur_span_str==span_str:
                        for j in range(cur_span.left, cur_span.right + 1):
                            examples[cur_span.inst_id].hp_labels[j] = "O"  
                            if j==cur_span.right: 
                                flag+=1
                                removal_annotations.append(span_str_with_type) 


        else:
            break
    new_all_spans=get_all_entity_spans(examples,"hp_labels")
    
    if add_false_positives:
        num_of_entity_tokens=np.sum(list(map(lambda span:span.right+1-span.left,new_all_spans)))
        num_false_positives=round(num_of_entity_tokens*config.false_positive_rate)
        add_false_positives_to_examples(config,examples,num_false_positives,new_all_spans)
        print(f'The number of false positive tokens:{num_false_positives}')
    assert flag==len(removal_annotations)
    assert len(new_all_spans)==len(all_spans)-flag
    return number_of_annotations,span_set,removal_annotations




def add_false_positives_to_examples(config,examples,num_false_positives,new_all_spans):
    all_label_type=list(config.label_map.values()) 
    all_label_type.remove('O')
    all_label_type.remove('_')
    all_token_span=list(map(lambda span:[Token(examples[span.inst_id].hp_labels[index],span.inst_id,index) for index in range(span.left,span.right+1)],new_all_spans))
    all_token_labels=[kk for span in all_token_span for kk in span ]
    selected_false_positives=np.random.choice(all_token_labels,num_false_positives,replace=False)
    for FP in selected_false_positives:
        candidate_labels=[kk for kk in all_label_type if kk !=FP.label]
        examples[FP.inst_id].hp_labels[FP.token_id]=random.choice(candidate_labels)#

