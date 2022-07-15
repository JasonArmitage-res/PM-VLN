"""
This file contains code from the following:
https://github.com/VegB/VLN-Transformer/blob/main/touchdown/agent.py

Source code used below is described in this paper:
"Multimodal Text Style Transfer for Outdoor Vision-and-Language Navigation"
https://arxiv.org/pdf/2007.00229.pdf

"""


import math
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from utils import _pm_inps, _mm_inps, trunc_norm


class BaseAgent:
    def __init__(self, env):
        self.env = env
        random.seed(1)

    def _vlntrans_sort_batch(self):
        """ getting sequence and corresponding lengths """
        # return self._get_tensor_and_length([item['instr_encoding'] for item in self.env.batch])
        input_ids, segment_ids, input_mask, sentence_ids, sentence_num = [], [], [], [], []
        if self.task_num == "t1_fl":
            batch = self.env.fl_batch
        else:
            batch = self.env.batch
        for item in batch:
            input_ids.append(item['encoder_input'][0])
            segment_ids.append(item['encoder_input'][1])
            input_mask.append(item['encoder_input'][2])
            sentence_ids.append(item['encoder_input'][3])
            sentence_num.append(item['encoder_input'][4])

        return torch.LongTensor(input_ids).to(self.device), \
               torch.LongTensor(segment_ids).to(self.device), \
               torch.LongTensor(input_mask).to(self.device), \
               torch.LongTensor(sentence_ids).to(self.device), sentence_num

    def _concat_textual_visual(self, textual_vectors, visual_vectors, sentence_nums, max_length):
        """
        :param textual_vectors: [batch_size, current_max_sent_num, hidden_dim]
        :param visual_vectors: [batch_size, n, hidden_dim], in which n is increasing
        :param sentence_nums: a list with batch_size elements
        :param max_length: an int
        :return: t_v_embeds, lengths, segment_ids
        """
        batch_size, pano_num, hidden_dim = visual_vectors.shape
        max_sent_num = textual_vectors.shape[1]
        t_v_embs, lengths, segment_ids = [], [], []
        for idx, sent_num in enumerate(sentence_nums):
            sent_num = sent_num if sent_num <= max_sent_num else max_sent_num
            pad_len = max_length - sent_num - pano_num
            cur_pano_num = pano_num
            if pad_len < 0:
                pad_len = 0
                cur_pano_num = max_length - sent_num
            pad_vec = torch.zeros(1, pad_len, hidden_dim).to(self.device)
            t_v_emb = torch.cat((textual_vectors[idx][:sent_num].unsqueeze(dim=0),
                                 visual_vectors[idx][-cur_pano_num:].unsqueeze(dim=0), pad_vec), dim=1)  # [1, max_length, hidden_dim]
            assert t_v_emb.shape[1] == max_length
            t_v_embs.append(t_v_emb)
            lengths.append(sent_num+cur_pano_num)
            seg_ids = torch.cat((torch.zeros(1, sent_num),
                                 torch.ones(1, cur_pano_num),
                                 torch.zeros(1, pad_len)), dim=1).long().to(self.device)
            segment_ids.append(seg_ids)

        return torch.cat(tuple(t_v_embs), dim=0), \
               torch.LongTensor(lengths).to(self.device), \
               torch.cat(tuple(segment_ids), dim=0).to(self.device)


class TaskFramework(BaseAgent):
    def __init__(self, opts, env, instr_encoder, pano_encoder, text_linear, model, fl_model, seg_emb):
        super(TaskFramework, self).__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.instr_encoder = instr_encoder
        self.text_linear = text_linear
        self.pano_encoder = pano_encoder
        self.model = model
        self.fl_model = fl_model
        self.seg_emb = seg_emb
        self.criterion = nn.CrossEntropyLoss(ignore_index=4)
        self.fl_criterion = nn.CrossEntropyLoss()

    def _encode_sentences(self, input_ids, segment_ids, sentence_ids):
        """input_ids, segment_ids, sentence_lengths: [batch_size, 512]"""
        input_length = (1 - (input_ids == 0).int()).sum(dim=1)
        encoded_word_embs, _ = self.instr_encoder(input_ids, input_length, segment_ids)  # [batch_size, max_sent_len, 768]
        max_sent_len = sentence_ids.max().item()
        encoded_sent_embs = []
        for i in range(max_sent_len):
            mask = (sentence_ids == i)  # [batch_size, 512]
            sent_len = mask.sum(dim=1).unsqueeze(dim=1)  # [batch_size]
            mask = mask.unsqueeze(-1).expand_as(encoded_word_embs).float()
            embs = encoded_word_embs.mul(mask).sum(dim=1)  # [batch_size, 768]
            sent_len = sent_len.expand_as(embs).float() + 1e-13  # [batch_size, 768]
            embs = embs.div(sent_len).unsqueeze(dim=1)
            encoded_sent_embs.append(embs)
        encoded_sent_embs = torch.cat(tuple(encoded_sent_embs), dim=1)
        return self.text_linear(encoded_sent_embs)  # [batch_size, current_max_sent_len, hidden_dim]

    def rollout(self, task_num, is_test):
        self.task_num = task_num
        if self.task_num == "t1_fl":
            trajs = self.env.reset()
            batch_size = len(self.env.fl_batch)
            max_route_len = 1
        else:
            trajs = self.env.reset()
            batch_size = len(self.env.batch)
            max_route_len = self.opts.max_route_len
        
        # Get batch for tx transformer
        input_ids, text_segment_ids, input_mask, sentence_ids, sentence_num = self._vlntrans_sort_batch()

        # Inputs for T1 FL
        if self.task_num == "t1_fl":
            hf_iids = []
            pano_cnt = []
            route_id_list = []
            tplan_cls = None
            for d in self.env.fl_batch:
                l = d["hf_iids"]
                r_id = d["wid"]
                hf_iids.append([l])
                route_id_list.append(r_id)
            r_TI = self.env._get_trajs(route_id_list)
            sent_iids_num = [len(x) for x in hf_iids]
            sent_iids_median = [int(math.ceil(i / 2)) for i in sent_iids_num]

            for i in range(batch_size):
                pano_cnt.append(1)
            in_id_in, a_m_in, t_t_pm = _pm_inps(hf_iids, sent_iids_num, pano_cnt, self.opts.max_window_len, tplan_cls)
            l_num_all = len(in_id_in) 
            in_id_pm = []
            a_m_pm = []
            l_rnd_list = random.choices([0, 1], k=l_num_all)
            r_seed = random.Random(500)
            label_list_t1 = []
            for idx, (l_i, l_am) in enumerate(zip(in_id_in, a_m_in)):
                l_i = l_i[0]
                l_am = l_am[0]
                l_seg_len = int(len(l_i) / 2)
                l_cur = idx
                label_t1 = r_seed.choice(l_rnd_list)
                l_alt_list = list(range(0, l_num_all))
                l_alt_list.remove(l_cur)
                l_alt = random.choice(l_alt_list)
                l_alt_full = in_id_in[l_alt][0]
                am_alt_full = a_m_in[l_alt][0]
                if label_t1 == 0:
                    l_new = [l_i[:l_seg_len] + l_alt_full[:l_seg_len]]
                    a_new = [l_am[:l_seg_len] + am_alt_full[:l_seg_len]]
                else:
                    l_new = [l_alt_full[:l_seg_len] + l_am[:l_seg_len]]
                    a_new = [am_alt_full[:l_seg_len] + l_am[:l_seg_len]]
                in_id_pm.append(l_new)
                a_m_pm.append(a_new)
                label_list_t1.append(label_t1)

        # Inputs for T2 VLN                
        else:
            hf_iids = [d["hf_iids"] for d in self.env.batch]
            pano_s = [d["route_panoids"] for d in self.env.batch]
            sent_iids_num = [len(x) for x in hf_iids]
            sent_iids_median = [int(math.ceil(i / 2)) for i in sent_iids_num]
            pano_cnt = [len(x) for x in pano_s]
            tplan_cls = None
            route_id_list = [d["route_id"] for d in self.env.batch]
            r_TI = self.env.get_trajs(route_id_list)

        ended = torch.BoolTensor([0] * batch_size).to(self.device)
        num_act_nav = [batch_size]

        encoded_texts = self._encode_sentences(input_ids, text_segment_ids, sentence_ids)  # [batch_size, cur_max_sent_len, hidden_dim]

        loss = 0
        total_steps = [0]
        pn_cur = 0
        encoded_panos = None

        for step in range(max_route_len):
            if self.task_num == "t1_fl":
                pano_s = []
                pano_names = [] 
                I = self.env._get_imgs()
                if step == 0:
                    pano_mean = [1] * batch_size
                for i in range(batch_size):
                    pano_s.append(['name%s' % i])
                    pano_names.extend(['name_%s' % i])
            else:
                if step == 0:
                    pano_mean = [34] * batch_size
                in_id_pm, a_m_pm, t_t_pm = _pm_inps(hf_iids, sent_iids_num, pano_mean, self.opts.max_window_len, tplan_cls)
                I, pano_names, pano_s = self.env.get_imgs() 

            pano_cur_list = []
            for l, p in zip(pano_s, pano_names):
                pns_idx = pano_s.index(l)
                pnc = pano_cnt[pns_idx] 
                if p in l:
                    pn_cur = l.index(p)
                else:
                    pn_cur = pn_cur
                if pn_cur >= pnc:
                    pn_cur = pnc-1
                pano_cur_list.append(pn_cur)
       
            input_ids_pm_pc = []
            input_mask_pm_pc = []
            text_segment_ids_pm_pc = []

            for idx, p in enumerate(pano_cur_list):
                a = in_id_pm[idx]
                if p >= len(a):
                    p_now = -1
                else:
                    p_now = p
                b = a[p_now]
                a1 = a_m_pm[idx]
                b1 = a1[p_now]
                a2 = t_t_pm[idx]
                b2 = a2[p_now]
                input_ids_pm_pc.append(b)
                input_mask_pm_pc.append(b1)
                text_segment_ids_pm_pc.append(b2)

            input_ids_pm_pc = torch.LongTensor(input_ids_pm_pc).to(self.device)
            input_mask_pm_pc = torch.LongTensor(input_mask_pm_pc).to(self.device)
            text_segment_ids_pm_pc = torch.LongTensor(text_segment_ids_pm_pc).to(self.device)

            vfl_I = I

            pano_mean_float = [float(x) for x in pano_mean]
            pano_mean_data = torch.Tensor(pano_mean_float)
            pano_mean_data = torch.unsqueeze(pano_mean_data, 1).to(self.device)

            input_ids_embs_pc = self.seg_emb(input_ids_pm_pc)

            I = self.pano_encoder(I).unsqueeze(1)  # [batch_size, 1, 256]           

            if encoded_panos is None:
                encoded_panos = I
            else:
                encoded_panos = torch.cat((encoded_panos, I), dim=1)

            t_v_embeds, lengths, segment_ids = \
                self._concat_textual_visual(encoded_texts, encoded_panos, sentence_num, self.opts.max_t_v_len)

            mm_I = vfl_I.squeeze(1)

            # Call main model
            model_outputs, _ = self.model(inputs_embeds=t_v_embeds, visual_embeds=mm_I)
            enc_outputs = model_outputs.hidden_states
            # Call FL_PM framework
            vb_logits, preds, cls_logits, vl_cls_logits, cls_lab = self.fl_model(input_ids_pc=input_ids_pm_pc, attention_mask_pc=input_mask_pm_pc, token_type_ids_pc=text_segment_ids_pm_pc, visual_embeds=vfl_I, tr_embeds=enc_outputs, input_ids_embs_pc=input_ids_embs_pc, task_num=self.task_num, r_maps=r_TI)
            mean_out = torch.argmax(cls_logits.detach(), dim=1).tolist()
            
            # Trajectory plans
            mean_out = [x for x in mean_out]
            mean_out = [1 if x==0 else x for x in mean_out]
            pano_mean = mean_out
            tplan_tuples = [*zip(sent_iids_num, sent_iids_median, mean_out)]
            tplan_cls = []
            for t in tplan_tuples:
                if t[0] < 2:
                    tn_out = [1] * t[2]
                else:
                    tn_out = trunc_norm(t)
                tplan_cls.append(tn_out)

            # Get actions
            # T1 FL
            if self.task_num == "t1_fl":
                pmf_preds = cls_lab.double().to(self.device)
                pmf_label = torch.LongTensor(label_list_t1).to(self.device)
                if is_test:
                    self.env.action_select(vb_logits, ended, num_act_nav, trajs, total_steps)
                    target = self.env.get_fl_class(is_test)
                    target_ = target.masked_fill(ended, value=torch.tensor(286))
                else:
                    target = self.env.get_fl_class(is_test)
                    target_ = target.masked_fill(ended, value=torch.tensor(286))
                    loss_1 = self.fl_criterion(vl_cls_logits, pmf_label)
                    loss_2 = self.fl_criterion(vb_logits, target_)
                    loss_comb = loss_1 + loss_2
                    loss += loss_comb * num_act_nav[0]                   
                    self.env.fl_env.action_step(target, ended, num_act_nav, trajs, total_steps)
                    target.unsqueeze(1)
            
            # T2 VLN
            else:
                if is_test:
                    self.env.action_select(vb_logits, ended, num_act_nav, trajs, total_steps)
                else:
                    target = self.env.get_gt_action(is_test)
                    target_ = target.masked_fill(ended, value=torch.tensor(4)) 
                    loss += self.criterion(vb_logits, target_) * num_act_nav[0]
                    self.env.env.action_step(target, ended, num_act_nav, trajs, total_steps)
                    target.unsqueeze(1)
            if not num_act_nav[0]:
                break
        loss /= total_steps[0]

        if self.task_num == "t1_fl":
            return trajs, pmf_label, pmf_preds, target_, loss, preds
        else:
            return trajs, loss, preds