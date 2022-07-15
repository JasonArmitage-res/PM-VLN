"""
This file comtains code from the following:
https://github.com/VegB/VLN-Transformer/blob/main/touchdown/trainer.py

Related methods described in this paper:
"Multimodal Text Style Transfer for Outdoor Vision-and-Language Navigation"
https://arxiv.org/pdf/2007.00229.pdf

"""


import math
import time
import torch
from utils import load_datasets, AverageMeter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TaskTrainer:
    def __init__(self, opts, agent, optimizer, bert_optimizer=None, v_bert_optimizer=None):
        self.opts = opts
        self.agent = agent
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    def train(self, epoch, task_num, tb_logger=None, task_env=None):
        self.task_num = task_num
        self.epoch = epoch
        if self.task_num == "t1_fl":
            self.agent.env = task_env
        else:
            self.agent.env = task_env
        self.agent.model.train()
        self.agent.instr_encoder.train()
        self.agent.fl_model.train()
        self.agent.seg_emb.train()
        if self.opts.model == 'vbforvln':
            self.agent.text_linear.train()
            self.agent.pano_encoder.train()
        self.agent.env.reset_epoch()

        losses = AverageMeter()
        batch_time = AverageMeter()

        end = time.time()
        if self.task_num == "t1_fl":
            self.train_iters_epoch = math.ceil(len(task_env.datafl) / self.opts.fl_batch_size)
        else:
            self.train_iters_epoch = math.ceil(len(task_env.data) / self.opts.vln_batch_size) 
        for iter_ in range(1, self.train_iters_epoch + 1):
            if self.task_num == "t1_fl":
                _, _, _, pmf_preds, loss, preds = self.agent.rollout(task_num, is_test=False)
            else:
                _, loss, preds = self.agent.rollout(task_num, is_test=False)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            if self.task_num == "t1_fl":
                losses.update(loss.item(), len(self.agent.env.fl_batch))
            else:
                losses.update(loss.item(), len(self.agent.env.batch))
            end = time.time()

            if tb_logger and iter_ % 10 == 0:
                current_iter = iter_ + (epoch - 1) * self.train_iters_epoch
                tb_logger.add_scalar('train/loss_train', loss, current_iter)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
                epoch, iter_, self.train_iters_epoch, batch_time=batch_time,
                loss=losses), end='')
        if tb_logger:
            tb_logger.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            tb_logger.add_scalar('epoch/train/loss', losses.avg, epoch)


    # Evaluation
    def eval_(self, epoch, task_num, tb_logger=None, task_env=None):
        self.task_num = task_num
        print("Evaluation loop starts")
        if self.task_num == "t1_fl":
            self.agent.env = task_env
            val_env = task_env
            phase = val_env.fl_env.name
            print('Evaluating on {} env ...'.format(phase))
        else:
            val_env = task_env
            self.agent.env = val_env
            phase = val_env.env.name
            print('Evaluating on {} env ...'.format(phase))
        losses = AverageMeter()
        batch_time = AverageMeter()
        ACC_list = []

        self.agent.env.reset_epoch()
        self.agent.model.eval()
        self.agent.instr_encoder.eval()
        self.agent.seg_emb.eval()
        self.agent.fl_model.eval()
        if self.opts.model == 'vbforvln':
            self.agent.text_linear.eval()
            self.agent.pano_encoder.eval()

        if self.task_num == "t1_fl":
            val_iters_epoch = math.ceil(len(val_env.datafl) / self.opts.fl_batch_size)
        else:
            val_iters_epoch = math.ceil(len(val_env.data) / self.opts.vln_batch_size)

        if self.task_num == "t1_fl":
            metrics = [0] * 1  # [TC]
        else:
            metrics = [0] * 3  # [TC, SPD, SED]
        with torch.no_grad():
            end = time.time()
            for iter_ in range(1, val_iters_epoch + 1):
                if self.task_num == "t1_fl":
                    trajs, pmf_label, pmf_preds, target_, loss, preds = self.agent.rollout(task_num, is_test=True)
                else:
                    trajs, loss, preds  = self.agent.rollout(task_num, is_test=True)
                if self.task_num == "t1_fl":
                    self.agent.env.eva_metrics(trajs, metrics, pmf_preds, pmf_label, preds, target_)
                else:
                    self.agent.env.eva_metrics(trajs, metrics)

                batch_time.update(time.time() - end)
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    epoch, iter_, val_iters_epoch, batch_time=batch_time))

        if self.task_num == "t1_fl":
            metrics = [m / len(val_env.datafl) for m in metrics]
        else:
            metrics = [m / len(val_env.data) for m in metrics]
            metrics = [m * 100 if m < 1 else m for m in metrics]
        if tb_logger:
            if self.task_num == "t1_fl":
                tb_logger.add_scalar('epoch/{}/ACC'.format(phase), metrics[0], epoch)
            else:
                tb_logger.add_scalar('epoch/{}/TC'.format(phase), metrics[0], epoch)
                tb_logger.add_scalar('epoch/{}/SPD'.format(phase), metrics[1], epoch)
                tb_logger.add_scalar('epoch/{}/SED'.format(phase), metrics[2], epoch)

        print("=======[%s] Evaluation Metrics=======" % phase)
        if self.task_num == "t1_fl":
                    print("ACC T1: %.2f" % tuple(metrics[:1]), end='')
        else:
            print("TC T2: %.2f, SPD: %.2f, SED: %.2f" % tuple(metrics[:3]), end='')
        print("================================")

        if self.task_num == "t1_fl":
            return metrics[0]
        else:
            return metrics[0], metrics[1]