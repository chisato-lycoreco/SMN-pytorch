import logging,os,argparse
from tensorboardX import SummaryWriter
import torch,random
import numpy as np

def init_logger(args=None):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args != None:
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            args.local_rank,
            args.device,
            args.n_gpu,
            bool(args.local_rank != -1),
            args.fp16,
        )
    return logger

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def checkoutput_and_setcuda(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    
    return args

def BasicConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        # required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
    )

    #限定最大对话轮次
    parser.add_argument(
        "--max_utter_num",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        default="./output/",
        type=str,
        # required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. ",
    )
    # parser.add_argument(
    #     "--train_file",
    #     default="train.txt",
    #     type=str,
    #     help="The input training file. If a data dir is specified, will look for the file there"
    #          + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    # )
    # parser.add_argument(
    #     "--dev_file",
    #     default="dev.txt",
    #     type=str,
    #     help="The input evaluation file. If a data dir is specified, will look for the file there"
    #          + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    # )
    # parser.add_argument(
    #     "--test_file",
    #     default="test.txt",
    #     type=str,
    #     help="The input test file.",
    # )
    parser.add_argument(
        "--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=80, type=int, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=80, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--max_seq_length",
        default=50,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument("--logging_steps", type=int, default=None, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every X updates steps.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")


    # infer case study
    parser.add_argument(
        "--infer_file",
        default="test.tsv",
        type=str,
        help="The input inference file.",
    )

    # learning rate and grad
    parser.add_argument("--lr_schedule_type", default="linear", type=str, help="lr_schedule_type")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Max gradient norm.")
    parser.add_argument("--sch", default='cos', type=str, help="Learning rate schedular.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")

    # Multi-GPU
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    # cache_dir and output_dir manager
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    # action choice
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_infer", action="store_true", help="Whether to run eval on the online case.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--do_lower_case", action="store_true", help="Whether to run lower case.")

    # fp16 manager
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
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    return parser

class Trainer(object):
    """
    Trainer
    """
    def __init__(self,
                 args,
                 model,
                 optimizer,
                 train_iter,
                 valid_iter,
                 logger,
                 valid_metric_name="-loss",
                 num_epochs=1,
                 save_dir=None,
                 log_steps=None,
                 valid_steps=None,
                 grad_clip=None,
                 lr_scheduler=None,
                 model_log=None,
                 save_summary=False):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger

        # self.generator = generator
        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.model_log = model_log
        self.save_summary = save_summary

        if self.save_summary:
            self.train_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "train"))
            self.valid_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "valid"))

        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.batch_num = 0

    def summarize_train_metrics(self, metrics, global_step):
        """ summarize_train_metrics """
        for key, val in metrics.items():
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, torch.Tensor):
                self.train_writer.add_scalar(key, val, global_step)

    def summarize_valid_metrics(self, metrics_mm, global_step):
        """ summarize_valid_metrics """
        for key in metrics_mm.metrics_cum.keys():
            val = metrics_mm.get(key)
            self.valid_writer.add_scalar(key, val, global_step)

    def train_epoch(self):
        """ train_epoch """
        raise NotImplemented

    def train(self):
        """ train """
        raise NotImplemented

    def save(self, is_best=False):
        """ save """
        raise NotImplemented

    def load(self, model_file, train_file):
        """ load """
        raise NotImplemented

    def init_message(self):
        self.train_start_message = "-" * 33 + " Model Training " + "-" * 33
        self.valid_start_message = "-" * 33 + " Model Evaulation " + "-" * 33