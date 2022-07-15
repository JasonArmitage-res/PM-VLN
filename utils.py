"""
This file contains code from the following:
https://github.com/VegB/VLN-Transformer/blob/main/touchdown/models/RConcat.py
https://github.com/VegB/VLN-Transformer/blob/main/touchdown/main.py

A theoretical description of the truncated normal distribution is provided in the following paper:
"The Truncated Normal Distribution"
https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

"""


import json, re, string
import numpy as np
import os, sys, torch
import warnings
from tensorboardX import SummaryWriter
import networkx as nx
import random
import shutil
import torch.utils.data as data
from collections import OrderedDict
from scipy.stats import truncnorm


base_vocab = ['<PAD>', '<START>', '<EOS>', '<UNK>']
padding_idx = 0


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def read_vocab(path):
    with open(path, encoding="utf-8") as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


class Tokenizer(object):
    """ Class to tokenize and encode a sentence. """
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, remove_punctuation=False, reversed=True, vocab=None, encoding_length=20):
        self.remove_punctuation = remove_punctuation
        self.reversed = reversed
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.table = str.maketrans({key: None for key in string.punctuation})
        self.word_to_index = {}
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        """ Break sentence into a list of words and punctuation """
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if
                     len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []

        splited = self.split_sentence(sentence)
        if self.reversed:
            splited = splited[::-1]

        if self.remove_punctuation:
            splited = [word for word in splited if word not in string.punctuation]

        for word in splited:  # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])

        encoding.append(self.word_to_index['<EOS>'])
        encoding.insert(0, self.word_to_index['<START>'])

        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length - len(encoding))
        return np.array(encoding[:self.encoding_length])

    def encode_instructions(self, instructions):
        rst = []
        for sent in instructions.strip().split('. '):
            rst.append(self.encode_sentence(sent))
        return rst


def load_datasets(splits, opts=None):
    data = []
    for split in splits:
        if opts.street_pt==True:
            assert split in ['train', 'dev']
        else:
            assert split in ['train', 'test', 'dev']
        with open('%s/data/%s.json' % (opts.dataset, split)) as f:
            for line in f:
                data.append(json.loads(line))
    return data


def shortest_path(pano, Graph):
    dis = {}
    queue = []
    queue.append([Graph.graph.nodes[pano], 0])
    while queue:
        cur = queue.pop(0)
        cur_node = cur[0]
        cur_dis = cur[1]
        if cur_node.panoid not in dis.keys():
            dis[cur_node.panoid] = cur_dis
            cur_dis += 1
            for neighbors in cur_node.neighbors.values():
                queue.append([neighbors, cur_dis])
                
    with open("path/"+pano+".json", "a") as f:
        json.dump(dis, f)

        
def resume_training(opts, model, instr_encoder, text_linear, fl_model, pano_encoder, seg_emb, optimizer, bert_optimizer=None, v_bert_optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opts.resume == 'latest':
        file_extention = '.pth.tar'
    elif opts.resume == 'SPD_best':
        file_extention = '_model_SPD_best.pth.tar'
    elif opts.resume == 'TC_best':
        file_extention = '_model_TC_best.pth.tar'
    elif opts.resume == 'ACC_best':
        file_extention = '_model_ACC_best.pth.tar'
    else:
        raise ValueError('Unknown resume option: {}'.format(opts.resume))
    exp_name = opts.resume_from if opts.resume_from is not None else opts.exp_name
    opts.resume = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, exp_name, file_extention))
    if os.path.isfile(opts.resume):
        checkpoint = torch.load(opts.resume, map_location=device)
        print("1 checkpoint", checkpoint)
        #print("2 checkpoint['model_state_dict']", checkpoint['model_state_dict'])
        #print("3 checkpoint['fl_model_state_dict']", checkpoint['fl_model_state_dict'])
        #model_pretrained_dict = OrderedDict()
        #for k, v in checkpoint['model_state_dict'].items():
            #name = k[7:] # remove `module.`
            #model_pretrained_dict[name] = v
        #print("4 model_pretrained_dict", model_pretrained_dict)        
        opts.start_epoch = checkpoint['epoch']
        #model.load_state_dict(model_pretrained_dict)
        model.load_state_dict(checkpoint['model_state_dict'])
        #fl_model_pretrained_dict = OrderedDict()
        #for k, v in checkpoint['fl_model_state_dict'].items():
            #name = k[7:] # remove `module.`
            #fl_model_pretrained_dict[name] = v
        #fl_model.load_state_dict(fl_model_pretrained_dict)
        fl_model.load_state_dict(checkpoint['fl_model_state_dict'])
        instr_encoder.load_state_dict(checkpoint['instr_encoder_state_dict'])
        seg_emb.load_state_dict(checkpoint['seg_emb_state_dict'])
        if opts.resume_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Optimizer resumed, lr = %f' % optimizer.param_groups[0]['lr'])
            #bert_optimizer.load_state_dict(checkpoint['bert_optimizer'])
            #v_bert_optimizer.load_state_dict(checkpoint['v_bert_optimizer'])
        if opts.model == 'vlntrans':
            text_linear.load_state_dict(checkpoint['text_linear_state_dict'])
            pano_encoder.load_state_dict(checkpoint['pano_encoder_state_dict'])
            if opts.resume_bert_optimizer:
                bert_optimizer.load_state_dict(checkpoint['bert_optimizer'])
                print('BERT Optimizer resumed, bert_lr = %f' % bert_optimizer.param_groups[0]['lr'])
                #v_bert_optimizer.load_state_dict(checkpoint['v_bert_optimizer'])
        try:
            best_SPD = checkpoint['best_SPD']
        except KeyError:
            print('best_SPD not provided in ckpt, set to inf.')
            best_SPD = float('inf')
        try:
            best_TC = checkpoint['best_TC']
        except KeyError:
            print('best_TC not provided in ckpt, set to 0.0.')
            best_TC = 0.0
        try:
            best_ACC = checkpoint['best_ACC']
        except KeyError:
            print('best_ACC not provided in ckpt, set to 0.0.')
            best_ACC = 0.0
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.resume, checkpoint['epoch']-1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(opts.resume))
    if opts.model == 'vlntrans':
        #return model, instr_encoder, text_linear, pano_encoder, optimizer, bert_optimizer, v_bert_optimizer, best_SPD, best_TC, best_ACC
        return model, instr_encoder, text_linear, fl_model, pano_encoder, optimizer, seg_emb, best_SPD, best_TC, best_ACC
    else:
        return model, instr_encoder, optimizer, best_SPD, best_TC


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    

def set_tb_logger(log_dir, exp_name, resume):
    """ Set up tensorboard logger"""
    log_dir = log_dir + '/' + exp_name
    # remove previous log with the same name, if not resume
    if not resume and os.path.exists(log_dir):
        import shutil
        try:
            shutil.rmtree(log_dir)
        except:
            warnings.warn('Experiment existed in TensorBoard, but failed to remove')
    return SummaryWriter(log_dir=log_dir)


def load_nav_graph(opts):
    with open("%s/graph/links.txt" % opts.dataset) as f:
        G = nx.Graph()
        for line in f:
            pano_1, _, pano_2 = line.strip().split(",")
            G.add_edge(pano_1, pano_2)        
    return G


def random_list(prob_torch, lists):
    x = random.uniform(0, 1)
    cum_prob = 0
    for i in range(len(lists) - 1):
        cum_prob += prob_torch[i]
        if x < cum_prob:
            return lists[i]
    return lists[len(lists) - 1]
    
    
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best_SPD, is_best_TC, is_best_ACC=None, epoch=-1):
    opts = state['opts']
    os.makedirs('{}/{}/{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name), exist_ok=True)
    filename = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '.pth.tar'))
    if opts.store_ckpt_every_epoch:
        filename = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '.%d.pth.tar' % epoch))
    torch.save(state, filename)
    if is_best_SPD:
        best_filename = (
            '{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '_model_SPD_best.pth.tar'))
        shutil.copyfile(filename, best_filename)
    if is_best_TC:
        best_filename = (
            '{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '_model_TC_best.pth.tar'))
        shutil.copyfile(filename, best_filename)

def save_checkpoint_fl(state, is_best_ACC, epoch=-1):
    opts = state['opts']
    os.makedirs('{}/{}/{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name), exist_ok=True)
    filename = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '.pth.tar'))
    if opts.store_ckpt_every_epoch:
        filename = ('{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '.%d.pth.tar' % epoch))
    torch.save(state, filename)
    if is_best_ACC:
        best_filename = (
            '{}/{}/{}/ckpt{}'.format(opts.checkpoint_dir, opts.model, opts.exp_name, '_model_ACC_best.pth.tar'))
        shutil.copyfile(filename, best_filename)


def input_img(pano, path):
    return np.load(path+"/"+pano+".npy")


class FLLoader(data.Dataset):
    def __init__(self, fl_json, fl_data):

        if fl_json == None:
            raise Exception('No data path specified.')

        self.h5f = fl_json

        self.ids = fl_data
        print("fl jsons imported")

    def __getitem__(self, index):
        print("get_item")
        instanceId = self.ids[index]
        print("instanceId", instanceId)


        #sys.exit("utils_2.py, line 267")

        # we force 50 percent of them to be a mismatch
        match = np.random.uniform() > self.mismatch if self.partition == 'train' else True

        target = match and 1 or -1

        if target == 1:  # load positive samples

            all_img = self.h5f[f'{instanceId}_images'][()]
            if self.partition == 'train':
                img = all_img[np.random.choice(range(all_img.shape[0]))]
            else:
                img = all_img[0]

            summaries = self.h5f[f'{instanceId}_summaries'][()]
            multi_wiki = summaries[np.random.choice(range(np.shape(summaries)[0]))]
            triple = self.h5f[f'{instanceId}_classes'][()]
            triple = triple[np.random.choice(range(triple.shape[0]))]
            cluster = self.h5f[f'{instanceId}_onehot'][()]

        else:
            # Negative samples are generated by picking random images
            # Other modalities remain unchanged
            all_idx = range(len(self.ids))
            rndindex = np.random.choice(all_idx)
            # random index to pick image ids at random
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index

            # load negative samples of images
            rndId = self.ids[rndindex]
            all_img = self.h5f[f'{rndId}_images']

            # negative samples are in train only
            if self.partition == 'train':
                img = all_img[np.random.choice(range(all_img.shape[0]))]
            else:
                img = all_img[0]
            
            # other modalities remain unchanged
            summaries = self.h5f[f'{rndId}_summaries'][()]
            multi_wiki = summaries[np.random.choice(range(np.shape(summaries)[0]))]
            triple = self.h5f[f'{instanceId}_classes'][()]
            triple = triple[np.random.choice(range(triple.shape[0]))]
            cluster = self.h5f[f'{instanceId}_onehot'][()]

        # output
        output = {
            'image': img,
            'multi_wiki': multi_wiki,
            'target': target,
            'cluster': cluster,
            'triple': triple
        }

        if self.partition != 'train':
            output['id'] = instanceId

        return output

    def __len__(self):
        return len(self.ids)


def sentence_tokenizer(tok_item): # newj
    sentence_ids = []
    sentence_num = 0
    d = ", 119,"
    for index, s in enumerate(str(tok_item[0]).split(d)):
        #print("index: ", index, " s: ", s)
        for idx, w in enumerate(s.split(", ")):
            #print("idx: ", idx, " w: ", w)
            if w == "[101":
                t = -1
                sentence_ids.append(t)
            elif w == " 102":
                t = -1
                sentence_ids.extend([t]*2)
            elif w == "0" and idx > 0:
                    t = -1
                    sentence_ids.append(t)
            elif w == "0]":
                    t = -1
                    sentence_ids.append(t)
            else:
                if idx == 0 and index > 0 and w != "0":
                    t = index
                    sentence_ids.extend([t]*2)
                else:
                    t = index
                    sentence_ids.append(t)
        sentence_num = index
    return sentence_ids, sentence_num


def _pm_inps(hf_iids, sent_iids_num, pano_mean, max_window_len, tplan_cls): # newj - move
    """
    Moving window over instructions.
    """
    in_id_pm, a_m_pm, t_t_pm = [], [], []
    mid_seg_len = int(max_window_len / 2)
    n_iter = 0
    for idx, i in enumerate(sent_iids_num):
        in_id_f, a_m_f, t_t_f = [], [], []
        if i > 1:
            if tplan_cls is not None:
                tn_l = tplan_cls[idx]
            else:
                tn_l = [1] * pano_mean[idx]
        else:
            tn_l = [1] * pano_mean[idx]
        
        # Pano step in route
        for step in range(pano_mean[idx]):
            iid_x = hf_iids[idx]
            iid_x_len = len(iid_x)-1
            tn_l[:] = [iid_x_len if n > iid_x_len else n for n in tn_l]
            if len(tn_l) < step:
                tn_l.append(iid_x_len)

            # Instructions > 1 sentence
            if len(iid_x) > 1:
                # Start
                if tplan_cls is None:
                    in_id_x = []
                    for i in range(2):
                        temp_in_id = []
                        temp_in_id = iid_x[0]+[119]+iid_x[1]
                        temp_in_id = temp_in_id[:mid_seg_len-3]
                        temp_in_id.insert(0, 101)
                        temp_in_id.extend((119, 102))
                        zer_os = mid_seg_len - len(temp_in_id)
                        temp_in_id = temp_in_id + [0] * (zer_os)
                        in_id_x.extend(temp_in_id)
                # End
                elif step == pano_mean[idx]-1: 
                    in_id_x = []
                    gt_sent_num = tn_l[step-1]
                    for i in range(2):
                        temp_in_id = []
                        if gt_sent_num >= len(iid_x):
                            gt_sent_num-=1
                            temp_in_id = iid_x[gt_sent_num-1]+[119]+iid_x[gt_sent_num]
                        elif gt_sent_num == 0:
                            temp_in_id = iid_x[gt_sent_num]+[119]+iid_x[gt_sent_num+1]
                        else:
                            temp_in_id = iid_x[gt_sent_num]
                        temp_in_id = temp_in_id[:mid_seg_len-3]
                        temp_in_id.insert(0, 101)
                        temp_in_id.extend((119, 102))
                        zer_os = mid_seg_len - len(temp_in_id)
                        temp_in_id = temp_in_id + [0] * (zer_os)
                        in_id_x = temp_in_id + in_id_x
                # Other
                else:
                    if len(iid_x) > 2:
                        iid_cnt = tn_l[step-1] - 1
                    else:
                        iid_cnt = tn_l[step-1]
                    in_id_x = []                
                    for i in range(2):
                        temp_in_id = []
                        temp_in_id.append(101)
                        iid_cnt = min(iid_cnt, iid_x_len)
                        while iid_cnt <= iid_x_len and len(temp_in_id) <= (mid_seg_len):
                            temp_in_id.extend(iid_x[iid_cnt])
                            temp_in_id.append(119)
                            iid_cnt+=1
                        temp_in_id = temp_in_id[:mid_seg_len-2]
                        temp_in_id.extend((119, 102))
                        zer_os = mid_seg_len - len(temp_in_id)
                        temp_in_id = temp_in_id + [0] * (zer_os)
                        in_id_x.extend(temp_in_id)
            
            # Instructions of 1 sentence
            else:
                in_id_x = [] 
                for i in range(2):
                    temp_in_id = []
                    temp_in_id.append(101)
                    if len(iid_x[0]) < mid_seg_len:
                        temp_in_id.extend(iid_x[0])
                    else:
                        if n_iter < (len(iid_x[0])-(mid_seg_len+n_iter)):
                            temp_in_id.extend(iid_x[0][n_iter:mid_seg_len+n_iter])
                            n_iter+=(mid_seg_len-3)
                        else:
                            temp_in_id.extend(iid_x[0][n_iter:mid_seg_len+n_iter])
                            n_iter+=(mid_seg_len-3)
                    temp_in_id.extend([0] * (mid_seg_len - len(temp_in_id)))
                    temp_in_id = temp_in_id[:(mid_seg_len - 2)]
                    temp_in_id.extend((119, 102))
                    in_id_x.extend(temp_in_id)
                n_iter = n_iter - (max_window_len - 6)
                n_iter += 2
           
            # Attention masks and token type IDs
            a_m_x = [1 if x > 0 else 0 for x in in_id_x]
            t_t_x = [0] * len(in_id_x)
            in_id_f.append(in_id_x)
            a_m_f.append(a_m_x)
            t_t_f.append(t_t_x)
            in_id_x = []
            iid_x = []
            in_id_x_len = 0
        in_id_pm.append(in_id_f)
        a_m_pm.append(a_m_f)
        t_t_pm.append(t_t_f)

    return in_id_pm, a_m_pm, t_t_pm

def trunc_norm(x):
    """
    A truncated Gaussian distribution function to generate trajectory plans.
    """
    sent, med, pano_mean = x
    s_v = truncnorm(a=1/med, b=sent/med, scale=med).rvs(size=pano_mean)
    sv_list = s_v.tolist()
    sv_int = [round(x) for x in sv_list]
    sv_sort = sorted(sv_int)
    return sv_sort


def _mm_inps(hf_iids, sent_iids_num, max_instr_len):
    """
    Full instruction inputs.
    """
    in_id_f, a_m_f, t_t_f = [], [], []
    n_iter = 0
    for idx, i in enumerate(sent_iids_num):
        iid_x = hf_iids[idx]
        temp_in_id = []
        for ins in iid_x:
            temp_in_id.append(101)
            temp_in_id.extend(ins)
            temp_in_id.append(119)
        if len(temp_in_id) >= (max_instr_len-2):
            temp_in_id = temp_in_id[:max_instr_len-2]
            temp_in_id.append(119)
        temp_in_id.append(102)
        zer_os = max_instr_len - len(temp_in_id)
        temp_in_id = temp_in_id + [0] * (zer_os)

        # Attention masks and token type IDs 
        a_m_x = [1 if x > 0 else 0 for x in temp_in_id]
        t_t_x = [0] * len(temp_in_id)
        in_id_f.append(temp_in_id)
        a_m_f.append(a_m_x)
        t_t_f.append(t_t_x)

    return in_id_f, a_m_f, t_t_f
