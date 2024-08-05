# Partial Annotation Learning for Biomedical Entity Recognition
This is the code base for paper [Partial Annotation Learning for Biomedical Entity Recognition](https://arxiv.org/abs/2305.13120).
We propose a TS-PubMedBERT-Partial-CRF partial annotation learning model and systematically study the effectiveness of partial annotation learning methods for biomedical entity recognition over different simulated scenarios of missing entity annotations. 
In this code base, we provide basic building blocks to allow arbitrary test of our model performance on five entity types and simulate real-world instances of the unlabeled entity problem with two distinct schemes to downsample the entities in the fully annotated dataset: Remove Annotations Randomly (RAR) and Remove All Annotations for Randomly Selected Surface Forms (RSFR). We also provide examples scripts for reproducing our results. In addition, we record all our the experimental details including all the hyperparameters and evaluation metrics, training loss etc. using wandb for your reference.
- `TS-PubMedBERT-Partial-CRF`: [TS-PubMedBERT-Partial-CRF](https://wandb.ai/possible961/TS-PubMedBERT-Partial-CRF)
- `PubMedBERT`: [PubMedBERT](https://wandb.ai/possible961/PubMedBERT)
- `BiLSTM-Partial-CRF`:[BiLSTM-Partial-CRF](https://wandb.ai/possible961/BiLSTM-Partial-CRF)
- `EER-PubMedBERT`:[EER-PubMedBERT](https://wandb.ai/possible961/EER-PubMedBERT)
- `Upper bond`: [Upper bond](https://wandb.ai/possible961/pubmedbert%20upperbond)
 


## Dependency
All Experiments were run on a an NVIDIA A100 GPU (40G) on Linux using python 3.7.

```
torch==1.13.0
transformers==4.4.2
allennlp==1.2.1
torch_struct==0.5
```
Install requirements

```
pip install -r requirements.txt
```


## File Structure

```
├── Scripts
│   └── train.sh
├── result_plot
│   ├── data
│   ├── confidence_interval_F1.ipynb
│   ├── confidence_interval_precision.ipynb
│   ├── confidence_interval_recall.ipynb
│   └── result_analysis_combine.ipynb
├── labels
│   ├── CellLine.txt
│   ├── Chemical.txt
│   ├── Disease.txt
│   ├── Gene.txt
│   └── Species.txt
├── data
│   ├── CellLine
│   ├── Chemical
│   ├── Disease
│   ├── Gene
│   └── Species
├── models
│   └── transformers
├── config.py
├── data_utils_self.py
├── model_utils.py
├── entity_removal.py
├── grammatical_transitions.py
└── run_self_training_ner.py

```

## Usage
### Core Files
- `run_self_training_ner.py`: main file to train the model
- `config.py`: Construct the arguments and some hyperparameters
- `data_utils_self.py`: read instances from file and convert to features
- `model_utils.py`: how to update the pseudo lables
- `entity_removal.py`: The implementation of RAR and RSFR scheme
- `grammatical_transitions.py`: construct the transition metrics for CRF model
- `train.sh`: The training script including arguments passing to the model
- `models/transformers/modeling_bert.py`: The PubmedBERT_Partial_CRF class contains the model architecture

### Hyperparameter Explaination
Here we explain hyperparameters using the script `train.sh`.

**Basic hyperparameters**
The blow hyperparameters doesn't need to change with the entity type and entity removal algorithm.

- `train_file`: The training dataset. It's `train_BIOUL.txt` for the whole experiments.
- `dev_file`: The evaluation dataset. It's `dev_BIOUL.txt` for the whole experiments.
- `test_file`: The test dataset. It's `test_BIOUL.txt` for the whole experiments.
- `model_type`: the type of model. It's `bert` for the whole experiments.
- `model_name_or_path`: the exact name for the bert-typed model. It's `pubmedbert-uncased` for the whole experiments.
- `weight_decay`: the weight decay (L2 regularization) applied to the model parameters to prevent overfitting. Default:`0`
- `adam_epsilon`: a small constant added to the denominator of the Adam optimizer update step to prevent any division by zero. Default:`1e-8`
- `adam_beta1`: the exponential decay rate for the first moment estimates in the Adam optimizer, which controls the momentum term. Default: `0.9`
- `adam_beta2`: the exponential decay rate for the second moment estimates in the Adam optimizer, which controls the variance term. Default: `0.98`
- `warmup_steps`: the number of steps for a linear warmup schedule. During warmup, the learning rate increases linearly from 0 to the initial learning rate before decaying according to the learning rate schedule. Default: `0`
- `per_gpu_train_batch_size`: the batch size per GPU/TPU core/CPU for the training phase. Default: `160` or `360` depending on the size of dataset.
- `per_gpu_eval_batch_size`: the batch size per GPU/TPU core/CPU for the evaluation phase. Default: `160` or `360` depending on the size of dataset.
- `logging_steps`: how often (in steps) training progress and metrics are logged.
- `save_steps`: how often (in steps) the model checkpoint is saved. Default: `100`
- `do_train`: a boolean flag indicating whether the training loop should be run. Default: `TRUE`
- `evaluate_during_training`: a boolean flag that, when set to True, enables evaluation of the model on the validation set during training at specified intervals. Default: `TRUE`
- `update_scheme`: the scheme or strategy used to update model parameters or training data. Default: `update_all`
- `output_dir`: the directory where model checkpoints, logs, and other outputs will be saved during and after training. Default: `output`
- `max_seq_length`:the maximum length (number of tokens) of the input sequences that will be processed by the model.  Default: `128`
- `overwrite_output_dir`: a boolean flag that, when set to True, allows the output directory specified by output_dir to be overwritten if it already exists.  Default: `TRUE`
- `data_cahce_index`: the directory where the processed data (e.g., tokenized inputs) will be cached to speed up subsequent runs. Default: `pubmedbert-uncased`
- `visible_device`: refers to the GPU device IDs that should be made visible to the program. Default: `0`
- `label_index`: the index in the dataset that contains the labels for supervised learning tasks. Default: `1`
- `dataset`: Default: `self`

**Core hyperparameters**
The blow hyperparameters varies with entity type and entity removal algorithm
- `data_dir`:  The data folder including the dataset to for model training or predicttion. It can be specified with `CellLine,Chemical,Disease,Gene,Species`.
- `labels`: The file that contains the entity categories. It can be specified with `CellLine.txt,Chemical.txt,Disease.txt,Gene.txt,Species.txt`
- `self_training_begin_step`: the training step at which self-training should begin.
- `self_training_period`: the period or frequency (in steps) at which self-training updates.
- `num_train_epochs`: the total number of times the training loop will iterate over the entire training dataset.
- `learning_rate`:  the learning rate at which the model's parameters are updated during training.
- `seed`: a random seed used to ensure reproducibility of results by controlling the random number generator.
- `entity_ratio`: the expected entity ratio
- `entity_ratio_margin`: a margin or threshold for the entity ratio
- `entity_removal_method`:  the method used to remove entities from the training data. It can be specified with `remove_annotations_randomly, remove_surfaceforms_randomly`
- `entity_removal_rate`: the rate at which entities are removed during training.
- `self_training_loss_weight`: the weight assigned to the loss from self-training examples during the training process.
- `prior_loss_weight`: the weight of margin-based marginal entity tag ratio loss on tagging posterior. 
- `overall_eer_weight`:  a weight applied to the overall expected entity ratio loss during model evaluation or training.


### Full experimental results
Full experimental results are documented in spreadsheet partial-annotation-learning-main/result_plot/data/final_results.csv and PDF 