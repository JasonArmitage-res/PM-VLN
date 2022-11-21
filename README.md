# A Priority Map for Vision-and-Language Navigation with Trajectory Plans and Feature-Location Cues

This repository contains source code and sample data for the priority map module (PM-VLN) and feature-location framework (FL<sub>PM</sub>) introduced in the paper "A Priority Map for Vision-and-Language Navigation with Trajectory Plans and Feature-Location Cues". Our PM-VLN module is inspired by priority maps - a mechanism described in neurophysiological research that modulates sensory processing on cues from the environment. We propose a computational implementation of cross-modal prioritisation that combines high-level trajectory estimation and feature-level localisation to optimise the prediction of actions in Vision-and-Language Navigation (VLN). The PM-VLN is integrated into the FL<sub>PM</sub> to enhance the performance a cross-modal transformer-based main model. In experiments, the FL<sub>PM</sub> with PM-VLN doubles the task completion rates of standalone transformers on the Touchdown benchmark.

<br/>

![system](/fig_flpm.png)

### Requirements
Python version >= 3.7

PyTorch version >= 1.8.0

``` bash
# clone the repository
git clone https://github.com/JasonArmitage-res/PM-VLN.git
cd PM-VLN
pip install -r requirements.txt
```

### Running the Framework
#### Steps to Run with Sample Data
This repository contains data samples to run the FL<sub>PM</sub> framework with integrated PM-VLN for initial review of the code. Please see below for details on the sample data. We tested and ran the framework with these samples on a single Tesla GPU with 16GB of RAM.   

- CD into the main directory.
- Install requirements for the project: ```pip install -r requirements.txt ```
- Unzip the archive containing the datasets (see below for details on contents): ```unzip SupMat_and_datasets_paper_id_672```
- Start training using the command line below.

``` bash
python main.py --dataset vln_sl_sample --img_feat_dir ./datasets/vln_sl_sample/features/ --pt_feat_dir ./datasets/vln_sl_sample/pt_features/ --hidden_dim 256 --model vbforvln --vln_batch_size 2 --fl_batch_size 5 --max_num_epochs 1 --exp_name train_sample_new --store_ckpt_every_epoch True --fl_dir datasets/mc_10_sample --fl_dataset mc_10 --fl_feat_dir datasets/mc_10_sample/features --fl_pt_feat_dir datasets/mc_10_sample/pt_features --max_instr_len 180 --max_window_len 80 --max_t_v_len 140 > flpm_sample_out.txt
```

#### Steps to Run with Full Datasets
Please see below information on the full auxiliary datasets and access to Touchdown. Notes on conducting individual experiments are provided in the paper. A sample command line for training the framework is added below (please update directory paths with locations of the MC-10 and Touchdown datasets).

``` bash
python main.py --dataset touchdown --img_feat_dir ./datasets/touchdown/features/ --pt_feat_dir ./datasets/touchdown/pt_features/ --hidden_dim 256 --model vbforvln --vln_batch_size 30 --fl_batch_size 60 --max_num_epochs 80 --exp_name train_new --store_ckpt_every_epoch True --fl_dir datasets/mc_10 --fl_dataset mc_10 --fl_feat_dir datasets/mc_10/features --fl_pt_feat_dir datasets/mc_10/pt_features --max_instr_len 180 --max_window_len 80 --max_t_v_len 140 > flpm_full_out.txt
```

### Data
#### Sample Data
We present in this repository samples from the following datasets:
  - mc_10_sample - data from MC-10 to train the PM-VLN module.
  - vln_sl_sample - sample VLN dataset containing nine modified samples of routes from StreetLearn.
#### Full Datasets
Experiments conducted for the paper above require pretraining the PM-VLN module on the auxiliary datasets generated for this research and evaluating the framework on the Touchdown benchmark.

Components in the PM-VLN module are pretrained on the following:
  - MC-10 - dataset of visual, textual and geospatial data for landmarks in 10 US cities.
  - TR-NY-PIT-central - set of image files graphing path traces for trajectory plan estimation in Manhattan and Pittsburgh. 

Versions of the auxiliary datasets are made available under Creative Commons public license at this [link](https://zenodo.org/record/6891965#.YtwoS3ZBxD8).

In order to access and download Touchdown and StreetLearn, please refer to this [link](https://sites.google.com/view/streetlearn/touchdown).

### Cite
Please use our code and add a citation if you find it interesting.

Armitage, Jason, Leonardo Impett, and Rico Sennrich. "A Priority Map for Vision-and-Language Navigation with Trajectory Plans and Feature-Location Cues." arXiv preprint arXiv:2207.11717 (2022). <br/>
Link: [https://arxiv.org/abs/2207.11717](https://arxiv.org/abs/2207.11717)

@article{armitage2022priority,
  title={A Priority Map for Vision-and-Language Navigation with Trajectory Plans and Feature-Location Cues},
  author={Armitage, Jason and Impett, Leonardo and Sennrich, Rico},
  journal={arXiv preprint arXiv:2207.11717},
  year={2022}
}

### License
This research is released under MIT license (please click [here](https://github.com/JasonArmitage-res/PM-VLN/blob/main/LICENSE)).
