"""
This file contains code from the following:
https://github.com/VegB/VLN-Transformer/blob/main/touchdown/main.py

Source code used below is described in this paper:
"Multimodal Text Style Transfer for Outdoor Vision-and-Language Navigation"
https://arxiv.org/pdf/2007.00229.pdf

"""


# Packages
import argparse
import os 
import texar.torch as tx
import torch
import torch.nn as nn
import torch.utils.data as data
# Project files
from packages.transformers_pm_vln import VisualBertConfig, VisualBertModel, VBforVLN
from packages.transformers_pm_vln import BertTokenizer, AdamW
from args import get_parser
from environments.t1_env import T1Batch
from environments.t2_env import load_features, T2Batch
from agent import TaskFramework
from trainer import TaskTrainer
from utils.utils import setup_seed, read_vocab, Tokenizer, resume_training, set_tb_logger, save_checkpoint, save_checkpoint_fl, FLLoader
from utils.utils_fl_pm import SegEmb, tpm_load_ckpt
# Models
from modeling.fl_pm import FLpm
from modeling.cnn_text import Conv_net, Text_linear


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = get_parser()
opts = parser.parse_args()
opts.dataset = 'datasets/%s' % opts.dataset
setup_seed(opts.seed)


def main(opts, fl_model_pt_dict=None, model_pt_dict=None):        
    # Setup
    vocab_file = "%s/vocab/vlntrans_vocab.txt" % opts.dataset
    tok_tx = tx.data.BERTTokenizer(pretrained_model_name='bert-base-uncased',
                                    hparams={'vocab_file': vocab_file})
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tb_logger = set_tb_logger('{}/{}'.format(opts.log_dir, opts.model), opts.exp_name, opts.resume)  
    pretrained_dict = None
 
    # FL_PM framework
    if task_num == "t1_fl": 
        visual_bert_config = VisualBertConfig(
            vocab_size=30522,
            hidden_size=opts.hidden_dim,
            visual_embedding_dim=100,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_labels=286,
        )
    else:
        visual_bert_config = VisualBertConfig(
            vocab_size=30522,
            hidden_size=opts.hidden_dim,
            visual_embedding_dim=100,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_labels=4,
        )
    fl_model = FLpm(visual_bert_config).to(device)
    
    if not opts.test:    
        if task_num == "t2_vln":
            if opts.street_pt is False:
                fl_model.pm_vln.pmf.load_state_dict(fl_model_pt_dict)
                fl_model.maxout.fc2_0 = nn.Linear(in_features=512, out_features=4).to(device)
                fl_model.maxout.fc2_1 = nn.Linear(in_features=512, out_features=4).to(device)
                fl_model.visual_bert.cls = nn.Linear(in_features=256, out_features=4).to(device)

    if opts.exp_number is not None:
        opts.exp_name = opts.exp_name + '_' + opts.exp_number
    best_SPD, best_TC, best_ACC = float("inf"), 0.0, 0.0

    if task_num == "t1_fl":
        fl_env = T1Batch(opts, batch_size=opts.fl_batch_size, splits=['_train'], tokenizer=tokenizer, tok_tx=tok_tx, name="train") 
        fl_val_env = T1Batch(opts, batch_size=opts.fl_batch_size, splits=['_dev'], tokenizer=tokenizer, tok_tx=tok_tx, name="eval")

    else:
        features, img_size = load_features(opts.img_feat_dir)
        train_env = T2Batch(opts, features, img_size, batch_size=opts.vln_batch_size, seed=opts.seed,
                                splits=['train'], tokenizer=tokenizer, tok_tx=tok_tx, name="train")
        val_env = T2Batch(opts, features, img_size, batch_size=opts.vln_batch_size, seed=opts.seed,
                                splits=['dev'], tokenizer=tokenizer, tok_tx=tok_tx, name="eval")

    # Main model
    if task_num == "t1_fl": 
        model_config = VisualBertConfig(
            vocab_size=30522,
            hidden_size=opts.hidden_dim,
            visual_embedding_dim=100,
            num_hidden_layers=4,
            num_attention_heads=4,
        )
    else:
        model_config = VisualBertConfig(
            vocab_size=30522,
            hidden_size=opts.hidden_dim,
            visual_embedding_dim=100,
            num_hidden_layers=4,
            num_attention_heads=4,
        )
    model = VBforVLN(model_config).to(device)

    # Instantiate other models
    instr_encoder = tx.modules.BERTEncoder(pretrained_model_name='bert-base-uncased').to(device)
    text_linear = Text_linear(opts).to(device)
    pano_encoder = Conv_net(opts).to(device)
    seg_emb = SegEmb().to(device)

    # Optmisers
    other_params = list(pano_encoder.parameters()) + list(fl_model.parameters()) + list(model.parameters()) + list(seg_emb.parameters()) + list(text_linear.parameters())
    tpm_params = list(fl_model.pm_vln.pmtp.cnxt_cls.parameters())
    optimizer = AdamW(other_params, lr=opts.lr)
    optimizer_tpm = AdamW(tpm_params, lr=opts.lr)
    if not opts.test:
        ck_pt = "/home/cluster/jarmit/data/methods/pytorch/dataset_prep/files/cnxt_ft_sim/17-checkpoint.pt"
        cnxt_dict, optimizer_tpm, _ = tpm_load_ckpt(ck_pt, optimizer_tpm)
        fl_model.pm_vln.pmtp.cnxt_cls.load_state_dict(cnxt_dict)

    # Data parallelisation over multiple devices
    instr_encoder = nn.DataParallel(instr_encoder)
    text_linear = nn.DataParallel(text_linear)
    pano_encoder = nn.DataParallel(pano_encoder)
    seg_emb = nn.DataParallel(seg_emb)
    model = nn.DataParallel(model)
    fl_model = nn.DataParallel(fl_model)

    # Define agent
    if task_num == "t1_fl":
        agent = TaskFramework(opts, fl_env, instr_encoder, pano_encoder, text_linear, model, fl_model, seg_emb)
    else:    
        agent = TaskFramework(opts, train_env, instr_encoder, pano_encoder, text_linear, model, fl_model, seg_emb)

    # Define trainer
    if task_num == "t1_fl":
        trainer = TaskTrainer(opts, agent, optimizer)
    else:
        trainer = TaskTrainer(opts, agent, optimizer) 

    if opts.resume:
        model, instr_encoder, text_linear, fl_model, pano_encoder, optimizer, seg_emb, best_SPD, best_TC, best_ACC = \
            resume_training(opts, model, instr_encoder, text_linear, fl_model, pano_encoder=pano_encoder, seg_emb=seg_emb,
                            optimizer=optimizer)


    # Evaluation on test set
    if opts.test:
        print("main 233 - is_test")
        assert opts.resume, 'The model was not resumed.'
        if task_num == "t1_fl":
            T1Batch(opts, batch_size=opts.fl_batch_size, seed=opts.seed, splits=['test'], tokenizer=tokenizer, name='test')
        else:
            test_env = T2Batch(opts, features, img_size, batch_size=opts.vln_batch_size, seed=opts.seed,
                                    splits=['test'], tokenizer=tokenizer, tok_tx=tok_tx, name='test')
        epoch = opts.start_epoch - 1

        if task_num == "t1_fl":
            trainer.eval_(epoch, task_num, tb_logger, task_env=fl_val_env)
        else:
            trainer.eval_(epoch, task_num, tb_logger, task_env=val_env)
            trainer.eval_(epoch, task_num, tb_logger, task_env=test_env)
        return

    # Training loop
    for epoch in range(opts.start_epoch, opts.max_num_epochs + 1):
        if task_num == "t1_fl":
            trainer.train(epoch, task_num, tb_logger, task_env=fl_env)
        else:
            trainer.train(epoch, task_num, tb_logger, task_env=train_env)
        
        # Evaluation
        if epoch % opts.eval_every_epochs == 0:
            if task_num == "t1_fl":
                ACC = trainer.eval_(epoch, task_num, tb_logger=tb_logger, task_env=fl_val_env)
                is_best_ACC = ACC >= best_ACC
                best_ACC = max(ACC, best_ACC)
                print("--> Best dev ACC: {}".format(best_ACC))
                if model is not None:
                    ckpt = {
                        'opts': opts,
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'fl_model_state_dict': fl_model.state_dict(),
                        'instr_encoder_state_dict': instr_encoder.state_dict(),
                        'seg_emb_state_dict': seg_emb.state_dict(), 
                        'best_ACC': best_ACC,
                        'optimizer': optimizer.state_dict()
                    }
                else:
                    ckpt = {
                        'opts': opts,
                        'epoch': epoch + 1,
                        'fl_model_state_dict': fl_model.state_dict(),
                        'instr_encoder_state_dict': instr_encoder.state_dict(),
                        'seg_emb_state_dict': seg_emb.state_dict(),
                        'best_ACC': best_ACC,
                        'optimizer': optimizer.state_dict()
                    }
            else: 
                TC, SPD = trainer.eval_(epoch, task_num, tb_logger, task_env=val_env)
                is_best_SPD = SPD <= best_SPD
                best_SPD = min(SPD, best_SPD)
                is_best_TC = TC >= best_TC
                best_TC = max(TC, best_TC)
                print("--> Best dev SPD: {}, best dev TC: {}".format(best_SPD, best_TC))
                if model is not None:
                    ckpt = {
                        'opts': opts,
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'fl_model_state_dict': fl_model.state_dict(),
                        'instr_encoder_state_dict': instr_encoder.state_dict(),
                        'seg_emb_state_dict': seg_emb.state_dict(),
                        'best_TC': best_TC,
                        'optimizer': optimizer.state_dict()
                    }
                else:
                    ckpt = {
                        'opts': opts,
                        'epoch': epoch + 1,
                        'fl_model_state_dict': fl_model.state_dict(),
                        'instr_encoder_state_dict': instr_encoder.state_dict(),
                        'seg_emb_state_dict': seg_emb.state_dict(),
                        'best_TC': best_TC,
                        'optimizer': optimizer.state_dict()
                    }
            if model is not None:
                ckpt['pano_encoder_state_dict'] = pano_encoder.state_dict()
                ckpt['text_linear_state_dict'] = text_linear.state_dict()
            else:
                ckpt['pano_encoder_state_dict'] = pano_encoder.state_dict()
                ckpt['text_linear_state_dict'] = text_linear.state_dict()
            if task_num == "t1_fl":
                save_checkpoint_fl(ckpt, is_best_ACC, epoch=epoch)
                print("--> Finished training - Task 1")
            else:
                save_checkpoint(ckpt, is_best_SPD, is_best_TC, epoch=epoch)
    fl_model_pt_dict = fl_model.module.pm_vln.pmf.state_dict()
    model_pt_dict = model.state_dict()
    print("--> Finished training")
    return fl_model_pt_dict, model_pt_dict


if __name__ == "__main__":

    if not opts.test:   
        task_num = "t1_fl"
        print("Starting T1")
        opts.max_num_epochs = 1        
        fl_model_pt_dict, model_pt_dict = main(opts)

    task_num = "t2_vln"
    print("Starting T2")
    if not opts.test:
        opts.max_num_epochs = 1
        main(opts, fl_model_pt_dict, model_pt_dict)
        print("--> Finished training - Task 2")
    else:
        main(opts)