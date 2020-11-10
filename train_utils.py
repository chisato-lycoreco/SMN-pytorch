import os

import shutil

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from utils import Trainer



def cal_performance(preds, labels,n_options=10):
    '''
    calculate R10@k metric for multi-turn dialogue system:
    R10@1 R10@2 R10@5
    preds:(n_sample,n_options)
    '''
    assert len(preds) == len(labels)
    total_num=len(preds)#500000
    
    
    n_samples=total_num//n_options
    
    pred,label=torch.tensor(preds),torch.tensor(labels)
    pred=torch.softmax(pred,dim=1)[:,1]#p(y=1|x,w)
    
    label=label.view(n_samples,-1)#50000x10
    pred=pred.view(n_samples,-1)#50000x10
    label=torch.argmax(label,dim=1,keepdim=True)#50000x1 正确选项的编号

    top1_pred=torch.topk(pred,k=1,dim=-1)[1]#50000xk topk indices
    top2_pred=torch.topk(pred,k=2,dim=-1)[1]
    top5_pred=torch.topk(pred,k=5,dim=-1)[1]

    max_label_1=label.expand_as(top1_pred)#50000x1
    max_label_2=label.expand_as(top2_pred)#50000x2
    max_label_5=label.expand_as(top5_pred)#50000x5

    isin_1=torch.eq(max_label_1,top1_pred)#50000x1 bool
    isin_2=torch.eq(max_label_2,top2_pred)#50000x2
    isin_5=torch.eq(max_label_5,top5_pred)#50000x5

    R10_1=torch.sum(isin_1).item()/n_samples#float
    R10_2=torch.sum(isin_2).item()/n_samples
    R10_5=torch.sum(isin_5).item()/n_samples

    metrics = {
        "R10@1":R10_1, 
        "R10@2":R10_2,
        "R10@5":R10_5
        }
    return metrics


class trainer(Trainer):
    def __init__(self, args,model, optimizer, train_iter, eval_iter, logger, valid_metric_name="+R10@1", save_dir=None,
                 num_epochs=5, log_steps=None, valid_steps=None, grad_clip=None, lr_scheduler=None, save_summary=False):

        super().__init__(args, model, optimizer, train_iter, eval_iter, logger, valid_metric_name, num_epochs,
                         save_dir, log_steps, valid_steps, grad_clip, lr_scheduler, save_summary)
                         

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.logger = logger

        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = num_epochs
        self.save_dir = save_dir if save_dir else self.args.output_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.save_summary = save_summary
       
        
        if self.save_summary:
            self.train_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "train"))
            self.valid_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "valid"))

        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.global_step = 0
        self.init_message()

    def train_epoch(self):
        
        self.epoch += 1
        train_start_message = "Training Epoch - {}".format(self.epoch)
        self.logger.info(train_start_message)

        tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
        
        loss_func=nn.CrossEntropyLoss()

        for batch_id, batch in tqdm(enumerate(self.train_iter,1)):
            self.model.train()
            
            utterance=batch[0].to(self.args.device)
            len_utterance=batch[1].to(self.args.device)
            num_utterance=batch[2].to(self.args.device)
            response=batch[3].to(self.args.device)
            len_response=batch[4].to(self.args.device)
            properity=batch[5].to(self.args.device)

            pred = self.model(
            utterance,len_utterance,num_utterance,
            response,len_response)

            loss = loss_func(pred,properity)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.grad_clip)
            else:
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            tr_loss += loss.item()
            nb_tr_examples += response.size(0)
            nb_tr_steps += 1

            if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                self.global_step += 1

            if self.global_step % self.log_steps == 0:
                # logging_loss = tr_loss / self.global_step
                self.logger.info("the current train_steps is {}".format(self.global_step))
                self.logger.info("the current logging_loss is {}".format(loss.item()))

            if self.global_step % self.valid_steps == 0:
                self.logger.info(self.valid_start_message)
                self.model.to(self.args.device)
                metrics = evaluate(self.args, self.model, self.eval_iter, self.logger)
                cur_valid_metric = metrics[self.valid_metric_name]
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric
                if is_best:
                    self.best_valid_metric = cur_valid_metric
                self.save(is_best)
                self.logger.info("-" * 85 + "\n")

    def train(self):
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(self.train_iter) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(self.train_iter) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        self.logger.info(self.train_start_message)
        self.logger.info("Num examples = %d", len(self.train_iter))
        self.logger.info("Num Epochs = %d", self.num_epochs)
        self.logger.info("Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        self.logger.info(
            "Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        self.logger.info("Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("Total optimization steps = %d", t_total)
        self.logger.info("logger steps = %d", self.log_steps)
        self.logger.info("valid steps = %d", self.valid_steps)
        self.logger.info("-" * 85 + "\n")
        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.args.fp16_opt_level)
        #single-gpu training
        if self.args.n_gpu==1:
            self.model=self.model.to(self.args.device)
        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model.cuda(), device_ids=[0,1])

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True,
            )

        for _ in range(int(self.num_epochs)):
            self.train_epoch()

    def save(self, is_best=False, save_mode="best"):
        model_file_name = "{}_state_epoch_{}.model".format(self.args.fusion_type,self.epoch) if save_mode == "all" else "{}_state.model".format(self.args.fusion_type)
        model_file = os.path.join(
            self.save_dir, model_file_name)
        torch.save(self.model.state_dict(), model_file)
        self.logger.info("Saved model state to '{}'".format(model_file))

        train_file_name = "{}_state_epoch_{}.train".format(self.args.fusion_type,self.epoch) if save_mode == "all" else "{}_state.train".format(self.args.fusion_type)
        train_file = os.path.join(
            self.save_dir, train_file_name)
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer.state_dict(),
                       "settings": self.args}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(train_state, train_file)
        self.logger.info("Saved train state to '{}'".format(train_file))

        if is_best:
            best_model_file = os.path.join(self.save_dir,"{}_best.model".format(self.args.fusion_type))
            best_train_file = os.path.join(self.save_dir,"{}_best.train".format(self.args.fusion_type))
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}-{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, model_file, train_file):

        if self.args.n_gpu>1:
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True

        model_state_dict = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict)
        self.logger.info("Loaded model state from '{}'".format(model_file))

        train_state_dict = torch.load(
            train_file, map_location=lambda storage, loc: storage)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_metric = train_state_dict["best_valid_metric"]
        self.batch_num = train_state_dict["batch_num"]
        self.optimizer.load_state_dict(train_state_dict["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
            self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
        self.logger.info(
            "Loaded train state from '{}' with (epoch-{} best_valid_metric-{:.3f})".format(
                train_file, self.epoch, self.best_valid_metric))


def evaluate(args, model,eval_dataset, logger):
    eval_loss, nb_eval_steps = 0.0, 0
    labels, preds = None, None
    model.eval()
    loss_func=nn.CrossEntropyLoss()
    for i,batch in tqdm(enumerate(eval_dataset)):
        
        utterance=batch[0].to(args.device)
        len_utterance=batch[1].to(args.device)
        num_utterance=batch[2].to(args.device)
        response=batch[3].to(args.device)
        len_response=batch[4].to(args.device)
        properity=batch[5].to(args.device)
        
        with torch.no_grad():
            logits = model(
            utterance,len_utterance,num_utterance,
            response,len_response) 
            tmp_eval_loss = loss_func(logits, properity)
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = properity.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, properity.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
 
    metrics = cal_performance(preds, labels)

    for key in sorted(metrics.keys()):
        logger.info("  %s = %s", key.upper(), str(metrics[key]))
    return metrics


