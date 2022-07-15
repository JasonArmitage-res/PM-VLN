"""
This file contains code from the following:
https://github.com/VegB/VLN-Transformer/blob/main/touchdown/env.py

Related methods are described in this paper:
"TOUCHDOWN: Natural Language Navigation and Spatial Reasoning in Visual Street Environments"
https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_TOUCHDOWN_Natural_Language_Navigation_and_Spatial_Reasoning_in_Visual_Street_CVPR_2019_paper.pdf

"""


import os
from glob import glob
import json
import numpy as np
import random
import re
import torch
import torch.nn.functional as F
from base_navigator import BaseNavigator


class T1EnvBatch:
    def __init__(self, opts, name=None):
        self.opts = opts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
    
    def next_fl_batch(self, datafl, seed_fl, ix_fl, batch_size):
        self.fl_batch = datafl[ix_fl:ix_fl+batch_size]
        if len(self.fl_batch) < batch_size:
            random.shuffle(datafl)
        else:
            ix_fl += batch_size
        self.ix_fl = ix_fl

        return self.fl_batch, self.ix_fl
    
    def get_imgs(self, fl_batch):
        imgs_fl = []
        for i in self.fl_batch:
            wid = str(i['wid'])+"_"
            wid_path = os.path.join(self.opts.fl_feat_dir,wid+'*.npy') 
            for i in glob(wid_path):
                img = np.load(i)
            img = img.squeeze(2)
            img = img[np.newaxis, ...]
            imgs_fl.append(img)
        imgs_fl = np.array(imgs_fl)

        return torch.from_numpy(imgs_fl).to(self.device)

    def get_trajs(self, route_id_list):
        rm_out = []
        for i in route_id_list:
            rf_name = "r_id_" + str(i) + ".npy"
            r_load = os.path.join(self.opts.fl_pt_feat_dir, rf_name)
            r_np = np.load(r_load)
            rm_out.append(r_np)
        rm_out = np.array(rm_out)

        return torch.from_numpy(rm_out).to(self.device)

    def _get_fl_class(self, fl_batch, is_test):
        gt_action = []
        for i, item in enumerate(fl_batch):
            if is_test:
                gt_action.append(item['geo_class'])  
            else:
                gt_action.append(item['geo_class'])      
        gt_action = np.array(gt_action)

        return torch.from_numpy(gt_action).long().to(self.device)

    def _action_select(self, a_prob, ended, num_act_nav, trajs, total_steps, fl_batch):
        """Called during testing."""
        a = []
        for i in range(len(fl_batch)):
            if ended[i].item():
                a.append([40])
                continue
            action_index = a_prob[i].argmax()
            action = action_index.item()
            a.append([action])            
            total_steps[0] += 1

        return torch.LongTensor(a).to(self.device)

    def action_step(self, target, ended, num_act_nav, trajs, total_steps):
        total_steps[0] += 1

    def _eva_metrics(self, trajs, fl_batch, metrics, pmf_preds, pmf_label, preds, target_):
        for i, item in enumerate(fl_batch):
            target_list = []
            success = 0
            s_0 = 0
            s_1 = 0
            pmf_prd = pmf_preds[i]
            pmf_lab = pmf_label[i]
            pred = preds[i]
            targ = target_[i]
            if pmf_lab == pmf_prd:
                s_0 = 0.5 
            if pred == targ:
                s_1 = 0.5
            success = s_0 + s_1
            metrics[0] += success

class T1Batch:
    def __init__(self, opts, batch_size, seed=10, splits=["_train"], tokenizer=None, tok_tx=None, name=None):
        self.fl_env = T1EnvBatch(opts, name)
        self.opts = opts
        self.name = name
        self.fl_vals = []
        self.datafl = []

        for split in splits:
            with open('%s/%s%s.json' % (self.opts.fl_dir, self.opts.fl_dataset, split)) as f:
                fl_json = json.loads(f.read())
                for k, v in zip(fl_json, fl_json.values()):
                    sum_en = v['en']
                    geo_class = v['class']
                    hf_iids = v['hf_iids']
                    self.fl_vals.append({'wid':k, 'sum_en':sum_en, 'geo_class':geo_class, 'hf_iids':hf_iids})   

        for i, item in enumerate(self.fl_vals):
            new_item = dict(item)
            instr = new_item['sum_en']
            new_item["encoder_input"] = tok_tx.encode_text(text_a=instr, text_b=None, max_seq_length=self.opts.max_instr_len)
            self.datafl.append(new_item)
        self.seed_fl = seed
        random.seed(self.seed_fl)
        random.shuffle(self.datafl)
        self.ix_fl = 0
        self.batch_size = batch_size

    def _load_nav_graph(self):
        self.graph = load_nav_graph(self.opts)
        print("Loading navigation graph done.")

    # Generate next batch
    def _next_fl_batch(self):
        fl_batch, ix_fl = self.fl_env.next_fl_batch(self.datafl, self.seed_fl, self.ix_fl, self.batch_size)
        self.fl_batch = fl_batch
        self.ix_fl = ix_fl
    
    # Get next batch of images
    def _get_imgs(self):

        return self.fl_env.get_imgs(self.fl_batch)

    def _get_trajs(self, route_id_list):

        return self.fl_env.get_trajs(route_id_list)

    def reset(self, print_info=False):
        self._next_fl_batch()
        wids = []
        geo_class_label = []
        trajs = []
        if print_info:
            print(self.fl_batch[0]["sum_en"])
        for item in self.fl_batch:
            wids.append(item["wid"])
            geo_class_label.append(item["geo_class"])
            trajs.append([item["wid"]])

        return trajs

    def get_fl_class(self, is_test):

        return self.fl_env._get_fl_class(self.fl_batch, is_test)

    def action_select(self, a_t, ended, num_act_nav, trajs, total_steps):

        return self.fl_env._action_select(a_t, ended, num_act_nav, trajs, total_steps, self.fl_batch)

    # Reset epoch
    def reset_epoch(self):
        self.ix_fl = 0

    def eva_metrics(self, trajs, metrics, pmf_preds, pmf_label, preds, target_):
        self.fl_env._eva_metrics(trajs, self.fl_batch, metrics, pmf_preds, pmf_label, preds, target_)