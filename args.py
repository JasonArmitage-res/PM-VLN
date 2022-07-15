"""
This file contains code from the following:
https://github.com/VegB/VLN-Transformer/blob/main/touchdown/main.py

Source code used below is described in this paper:
"Multimodal Text Style Transfer for Outdoor Vision-and-Language Navigation"
https://arxiv.org/pdf/2007.00229.pdf

"""


import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vbforvln', type=str, choices=['vbforvln', 'vlntrans'])
    parser.add_argument('--dataset', default='', type=str, help='Path to VLN dataset.')
    parser.add_argument('--fl_dir', default='', type=str, help='Path to FL dataset.')
    parser.add_argument('--fl_dataset', default='', type=str, help='Select version of FL dataset.')
    parser.add_argument('--img_feat_dir', default='', type=str, help='Path to pre-cached image features.')
    parser.add_argument('--fl_feat_dir', default='', type=str, help='Path to pre-cached FL features.')
    parser.add_argument('--pt_feat_dir', default='', type=str, help='Path to VLN path trace features.')
    parser.add_argument('--fl_pt_feat_dir', default='', type=str, help='Path to FL path trace features.')
    parser.add_argument('--log_dir', default='tensorboard_logs/touchdown', type=str, help='Path to tensorboard log files.')
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str, help='Path to the checkpoint dir.')
    parser.add_argument('--resume', default='', type=str, choices=['latest', 'TC_best', 'SPD_best', 'ACC_best'])
    parser.add_argument('--resume_from', default=None, type=str, help='resume from other experiment')
    parser.add_argument('--store_ckpt_every_epoch', default=False, type=bool)
    parser.add_argument('--ckpt_epoch', default=-1, type=int)
    parser.add_argument('--test', default=False, type=bool, help='No training. Resume from a model and run testing.')
    parser.add_argument('--seed', default=10, type=int, help='random seed')
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--max_num_epochs', default=80, type=int, help='Max training epoch.')
    parser.add_argument('--vln_batch_size', default=30, type=int)
    parser.add_argument('--fl_batch_size', default=60, type=int)
    parser.add_argument('--eval_every_epochs', default=1, type=int, help='How often do we eval the trained model.')
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr', default=0.00025, type=float)
    parser.add_argument('--finetune_bert', default=False, type=bool)
    parser.add_argument('--bert_lr', default=0.00001, type=float)
    parser.add_argument('--resume_optimizer', default=False, type=bool)
    parser.add_argument('--resume_bert_optimizer', default=False, type=bool)
    parser.add_argument('--max_instr_len', default=180, type=int, help='Max instruction token num.')
    parser.add_argument('--max_window_len', default=80, type=int, help='Max length for PM-VLN module sequence inputs.')
    parser.add_argument('--street_pt', default=False, type=bool, help='Option for an additional pretraining VLN task.')
    parser.add_argument('--max_route_len', default=55, type=int, help='Max trajectory length.')
    parser.add_argument('--max_t_v_len', default=140, type=int,
                        help='Max length of the concatenation of sentence embeddings and trajectory embeddings.')
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--exp_name', default='experiments', type=str,
                        help='Name of the experiment. It decides where to store samples and models')
    parser.add_argument('--exp_number', default=None, type=str)
    parser.add_argument('--workers', default=0, type=int)
    opts = parser.parse_args()

    return parser