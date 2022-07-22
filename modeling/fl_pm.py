"""
Source code for the PM-VLN module and FL_PM framework presented in our paper:
"A Priority Map for Vision-and-Language Navigation with Trajectory Plans and Feature-Location Cues"

The ConvNeXt architecture is introduced in the following paper:
https://arxiv.org/pdf/2201.03545.pdf

VisualBERT is described in the following paper:
https://arxiv.org/pdf/1908.03557.pdf

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation
from packages.transformers_pm_vln import VisualBertModel, VisualBertPreTrainedModel, VBModelforFLrl, BertConfig
from utils.utils_fl_pm import maxout_mlp, ListModule, log2d, LoG2d, USMBase, AdaptiveUSM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate ConvNeXt Tiny model.
cnxt_cls = torchvision.models.convnext_tiny()
cnxt_cls.classifier[2] = nn.Linear(768, 66, bias=True)
cnxt_cls.to(device)



class PMTP(nn.Module):
    """
    Submodule G_PMTP for trajectory estimation in the PM-VLN module.
    """
    def __init__(self):
        super(PMTP, self).__init__()
        self.cnxt_cls = cnxt_cls

    def forward(self, x_rm):
        """
        Forward through the submodule.
        :param x_rm: (torch.tensor) Path trace.
        :return cls_logits: (torch.tensor) Logits for estimate of step count in tr_T.
        """
        rm_emb = x_rm
        self.cnxt_cls.eval()
        # Classification with ConvNeXt Tiny model.        
        cls_logits = self.cnxt_cls(x_rm)

        return cls_logits


class PMF(nn.Module):
    """
    Submodule G_PMF for feature-level localisation in the PM-VLN module.
    """
    def __init__(self):
        super(PMF, self).__init__()       
        self.v_pre = AdaptiveUSM(in_channels=1, in_side=100, kernel_size=3, cuda=True)       
        self.fc0 = nn.Linear(256, 100)
        self.fc1 = nn.Linear(80, 100)
        self.fc2 = nn.Linear(200, 100)
        # VL embeddings
        self.lstm = nn.LSTM(100, 512)
        self.cls = nn.Linear(512, 2)

    def forward(self, x, x_l):
        """
        Forward through the submodule.
        :param x: (torch.tensor) Visual input.
        :param x_l: (torch.tensor) Linguistic input.
        :return v_embeds: (torch.tensor) Visual embeddings after VBF.
        :return v_attention_mask: (torch.tensor) Visual attention masks.
        :return v_token_type_ids: (torch.tensor) Visual attention masks.
        :return vl_cls_logits: (torch.tensor) Predictions for localising spans.
        """
        # VBF on visual inputs
        usm_x = self.v_pre(x)
        v_embeds = usm_x.squeeze(1)
        v_attention_mask = torch.ones(v_embeds.shape[:-1], dtype=torch.long).to(torch.device('cuda'))
        v_token_type_ids = torch.ones(v_embeds.shape[:-1], dtype=torch.long).to(torch.device('cuda'))
        # Linguistic inputs
        x_l_0 = self.fc0(x_l)
        x_l_1 = self.fc1(x_l_0.permute(0, 2, 1))
        # Cross-modal embeddings
        x_conc = torch.cat((x_l_1, v_embeds), 2)
        vl_emb = self.fc2(x_conc)     
        # Linguistic inputs
        vl_emb_cnt = list(vl_emb.size())[0]
        vl_emb_p = vl_emb.permute(1, 0, 2)
        # Recurrent layer returns logits
        h0 = torch.zeros((1, vl_emb_cnt, 512), requires_grad=True).to(device)
        c0 = torch.zeros((1, vl_emb_cnt, 512), requires_grad=True).to(device)
        out_put, (hn, cn) = self.lstm(vl_emb_p, (h0, c0))
        vl_cls_logits = self.cls(hn[-1]).to(device)    

        return v_embeds, v_attention_mask, v_token_type_ids, vl_cls_logits


class PMVLN(nn.Module):
    """
    PM-VLN module for trajectory estimation and feature-level localisation.
    """
    def __init__(self, PMF, PMTP):
        super(PMVLN, self).__init__()
        self.pmtp = PMTP
        self.pmf = PMF

    def forward(self, x, x_l, x_rm):
        """
        Forward through the module.
        :params: see submodules.
        :return: see submodules.
        """
        v_embeds, v_attention_mask, v_token_type_ids, vl_cls_logits = self.pmf(x, x_l)
        self.pmtp.eval()        
        cls_logits = self.pmtp(x_rm)
        
        return v_embeds, v_attention_mask, v_token_type_ids, vl_cls_logits, cls_logits


class CFns(nn.Module):
    """
    Submodule G_CFns for combining outputs from the main model and cross-modal self-attention.
    """
    def __init__(self):
        super(CFns, self).__init__()
        self.fc0 = nn.Linear(in_features=1024, out_features=256)
        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu_af = nn.ReLU()

    def forward(self, x_pm, x_m):
        """
        Forward through the submodule.
        :param x_pm: (torch.tensor) Outputs from cross-modal self attention.
        :param x_m: (torch.tensor) Outputs from the main model.
        :return x_out: (torch.tensor) Combined representation for the inputs.
        """
        x_pm = self.relu_af(self.fc0(x_pm))
        x_m = self.relu_af(self.fc1(x_m))
        x_cat = torch.cat([x_pm, x_m], dim=2)
        x_out = self.relu_af(self.fc2(x_cat))
        
        return x_out


class FLpm(nn.Module):
    """
    FL_PM framework contains the PM-VLN and combines outputs ahead of predicting actions with maxout activation.
    """
    def __init__(self, visual_bert_config):
        super(FLpm, self).__init__()
        pmf = PMF()
        pmtp = PMTP()
        self.pm_vln = PMVLN(pmf, pmtp)
        self.num_labels = visual_bert_config.num_labels
        self.visual_bert = VBModelforFLrl(visual_bert_config)
        self.CFns = CFns()
        self.maxout = maxout_mlp(out_features=self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_attention_mask=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
        output_attentions=None,
        output_hidden_states=True, # Newj
        return_dict=None,
        labels=None,
        tr_embeds=None,
        step_data_pc=None,
        input_ids_pc=None,
        attention_mask_pc=None,
        token_type_ids_pc=None,
        input_ids_embs_pc=None,
        task_num=None,
        pano_mean_data=None,
        pad_panos=None,
        r_maps=None,
    ):
        """
        Forward through the framework.
        :param visual_embeds: (torch.tensor) Visual inputs.
        :param input_ids_pc: (torch.tensor) Linguistic inputs.
        :param attention_mask_pc: (torch.tensor) Linguistic attention masks.
        :param token_type_ids_pc: (torch.tensor) Linguistic token types.
        :param tr_embeds: (torch.tensor) Outputs from the main model.
        :param input_ids_embs_pc: (torch.tensor) Linguistic inputs.
        :param r_maps: (torch.tensor) Path traces.
        :return vb_logits: (torch.tensor) Action predictions.
        :return preds: (torch.tensor) Logits for actions predictions.
        :return vl_cls_logits: (torch.tensor) Predictions for localising spans.
        :return cls_lab: (torch.tensor) Logits for predictions.
        """
        # PM-VLN
        v_embeds, v_attention_mask, v_token_type_ids, vl_cls_logits, cls_logits = self.pm_vln(visual_embeds, input_ids_embs_pc, r_maps)

        # Span selection
        cls_lab = torch.argmax(vl_cls_logits, dim=1).float().to(device)    
        cls_out = torch.unsqueeze(cls_lab, dim=1).to(device)
        input_ids = torch.LongTensor(()).to(device)
        attention_mask = torch.LongTensor(()).to(device)
        token_type_ids = torch.LongTensor(()).to(device)
        # Input ids
        for c, t in zip(cls_out, input_ids_pc):
                if c == 0:
                    t_sel_a = torch.unsqueeze(t[:40], 0).to(device)
                    input_ids = torch.cat((input_ids, t_sel_a)).to(device)
                else:
                    t_sel_b = torch.unsqueeze(t[40:], 0).to(device)
                    input_ids = torch.cat((input_ids, t_sel_b)).to(device)
        # Attention masks
        for c, t in zip(cls_out, attention_mask_pc):
                if c == 0:
                    t_sel_a = torch.unsqueeze(t[:40], 0).to(device)
                    attention_mask = torch.cat((attention_mask, t_sel_a)).to(device)
                else:
                    t_sel_b = torch.unsqueeze(t[40:], 0).to(device)
                    attention_mask = torch.cat((attention_mask, t_sel_b)).to(device)
        # Token Type IDs
        for c, t in zip(cls_out, token_type_ids_pc):
                if c == 0:
                    t_sel_a = torch.unsqueeze(t[:40], 0).to(device)
                    token_type_ids = torch.cat((token_type_ids, t_sel_a)).to(device)
                else:
                    t_sel_b = torch.unsqueeze(t[40:], 0).to(device)
                    token_type_ids = torch.cat((token_type_ids, t_sel_b)).to(device)
        index_to_gather = attention_mask.sum(1) - 2
 
        # Call to VisualBERT model
        vbm_hidden_states = self.visual_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            visual_embeds=v_embeds,
            visual_attention_mask=v_attention_mask,
            visual_token_type_ids=v_token_type_ids,
            image_text_alignment=image_text_alignment,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Combine hidden states across layers
        hs_4 = torch.cat((tuple([vbm_hidden_states[i] for i in [-1, -2, -3, -4]])), dim=-1)
        tr_hs_4 = torch.cat((tuple([tr_embeds[i] for i in [-1, -2, -3, -4]])), dim=-1)
        hs_4 = F.pad(hs_4, pad=(0, 0, 0, 240 - hs_4.shape[1]))      
        # Combine outputs
        com_out = self.CFns(hs_4, tr_hs_4)
        # Classification with maxout activation
        index_to_gather = (
            index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1, com_out.size(-1))
        )
        pooled_output = torch.gather(com_out, 1, index_to_gather)
        logits = self.maxout(pooled_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        # Outputs
        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            log_softmax = nn.LogSoftmax(dim=-1)
            reshaped_logits = log_softmax(reshaped_logits)
            loss = loss_fct(reshaped_logits, labels.contiguous())
        preds = torch.argmax(reshaped_logits, dim=-1)
        preds = torch.flatten(preds)

        return reshaped_logits, preds, cls_logits, vl_cls_logits, cls_lab