
class Config:
    def __init__(self, args) -> None:
        """
        Construct the arguments and some hyperparameters
        :param args:
        """
        # Model hyper parameters
        self.model_type = args.model_type
        self.model_name_or_path = args.model_name_or_path
        self.config_name = args.config_name
        self.tokenizer_name = args.tokenizer_name
        self.cache_dir = args.cache_dir
        self.do_lower_case = args.do_lower_case
        self.not_use_best_model = args.not_use_best_model
        self.data_cahce_index = args.data_cahce_index
        self.evaluation_method=args.evaluation_method

        # Data specification
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.dataset = args.dataset
        self.labels= args.labels
        self.overwrite_output_dir = args.overwrite_output_dir
        self.overwrite_cache = args.overwrite_cache
        self.integration_method = args.integration_method
        self.label_index = args.label_index
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.shuffle_data=args.shuffle_data
        self.separator = args.separator
        self.cache_dir_name = args.cache_dir_name
        self.write_examples_to_file = args.write_examples_to_file
        self.load_big_file = args.load_big_file

        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2    
        self.max_grad_norm = args.max_grad_norm
        self.fp16_opt_level = args.fp16_opt_level
        self.local_rank=args.local_rank



        # Training hyperparameter
        self.max_seq_length = args.max_seq_length
        self.do_train = args.do_train
        self.do_eval = args.do_eval
        self.do_predict = args.do_predict
        self.evaluate_during_training = args.evaluate_during_training
        self.per_gpu_train_batch_size = args.per_gpu_train_batch_size
        self.per_gpu_eval_batch_size=args.per_gpu_eval_batch_size
        self.gradient_accumulation_steps=args.gradient_accumulation_steps
        self.no_cuda = args.no_cuda
        self.visible_device = args.visible_device    
        self.seed = args.seed
        self.fp16 = args.fp16
        self.fp16_opt_level = args.fp16_opt_level

        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.adam_epsilon = args.adam_epsilon
        self.max_grad_norm = args.max_grad_norm
        self.num_train_epochs = args.num_train_epochs
        self.max_steps = args.max_steps
        self.batch_size = args.batch_size
        self.warmup_steps = args.warmup_steps
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps
        self.eval_all_checkpoints = args.eval_all_checkpoints
        self.server_ip = args.server_ip
        self.server_port = args.server_port


        # wandb hyperparameter
        self.sweep_method=args.sweep_method
        self.sweep_round=args.sweep_round
        self.project_name=args.project_name
        self.sweep_file=args.sweep_file



        # mean teacher
        self.mt=args.mt
        self.mt_updatefreq=args.mt_updatefreq
        self.mt_class=args.mt_class
        self.mt_lambda=args.mt_lambda
        self.mt_rampup=args.mt_rampup
        self.mt_alpha1=args.mt_alpha1
        self.mt_alpha2=args.mt_alpha2
        self.mt_beta=args.mt_beta
        self.mt_avg=args.mt_avg
        self.mt_loss_type=args.mt_loss_type

        # virtual adversarial training
        self.vat=args.vat
        self.vat_eps=args.vat_eps
        self.vat_lambda=args.vat_lambda
        self.vat_beta=args.vat_beta
        self.vat_loss_type=args.vat_loss_type

        # self training
        self.whether_self_training=args.whether_self_training
        self.self_training_reinit=args.self_training_reinit
        self.self_training_begin_step=args.self_training_begin_step
        self.self_training_label_mode=args.self_training_label_mode
        self.self_training_period=args.self_training_period
        self.self_training_hp_label=args.self_training_hp_label
        self.self_training_hp_label_category=args.self_training_hp_label_category
        self.self_training_ensemble_label=args.self_training_ensemble_label
        self.whether_category_oriented=args.whether_category_oriented
        self.confidence_test=args.confidence_test
        self.update_every_period=args.update_every_period

        #  Use data from weak.json
        self.load_weak=args.load_weak
        self.remove_labels_from_weak=args.remove_labels_from_weak
        self.rep_train_against_weak=args.rep_train_against_weak

        # feature integration
        self.pos = args.pos
        self.lexicon = args.lexicon
        self.whether_integrate_pos = args.whether_integrate_pos
        self.whether_integrate_lexicon = args.whether_integrate_lexicon
        self.pos_embeds_size = args.pos_embeds_size
        self.lexicon_embeds_size = args.lexicon_embeds_size
