# Text-to-Image-Attentional-Generative-Adversarial-Networks

This repository is about assessing an Attentional Generative Adversarial Network (AttnGAN) method which generates images from text. This method can pay attention to the relevant words in the written natural language to synthesize details at
different sub-regions of the image. Due to hardware limitations, this report just evaluates and extends the AttnDCGAN model. To evaluate this method, some text descriptions defined in the original paper are used to compare the generated images by the AttnDCGAN over the CUB dataset. For extension, different kinds of words are tested in the text description to see the behavior of
the system in generating images.

This repo aims to utilize the proposed network in

T. Xu, P. Zhang, Q. Huang, H. Zhang, Z. Gan, X. Huang, and X. He. ”[Attngan: Fine-grained text to image generation with attentional
generative adversarial networks](https://arxiv.org/pdf/1711.10485).” In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1316-1324. 2018.

to generate high-resolution images from text descriptions. In this model, two networks are proposed for this purpose,
AttnGAN and AttnDCGAN. AttnGAN can generate three images with different resolutions, 64x64x3, 128x128x3 and 256x256x3. However, the AttnDCGAN can just generate the
64x64x3 image. Here, AttnGAN is built over the CUB and COCO dataset,s and AttnDCGAN is just built over the CUB dataset.

The code of the paper AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative
Adversarial which is implemented AttnGAN in Pytorch is provided in [here](https://github.com/taoxugit/AttnGAN).

## Installation
This project needs a list of packages. To begin with, a Conda environment is created to install these
main packages.
```
conda create --name AttnGAN python=2.7
conda activate AttnGAN
conda install pytorch torchvision -c pytorch
pip install python-dateutil easydict pandas torchfile scikit-image
conda install nltk
```

Then, clone the GitHub repository in the specific folder and download the requisite datasets and
pretrained models for running this network.
```
git clone https://github.com/taoxugit/AttnGAN.git
```
dataset
• Download our preprocessed metadata for birds coco and save them to data/
• Download the birds image data. Extract them to data/birds/
• Download coco dataset and extract the images to data/coco/
• Extract text.zip under data/birds
• Extract train2014-text.zip and val2014-text.zip under data/coco

Pretrained model
• DAMSM for bird. Download and save it to DAMSMencoders/
• DAMSM for coco. Download and save it to DAMSMencoders/
• AttnGAN for bird. Download and save it to models/
• AttnGAN for coco. Download and save it to models/
• AttnDCGAN for bird. Download and save it to models/

Here, AttnDCGAN is a variant of AttnGAN that applies the proposed attention mechanisms to the
DCGAN framework.
Then the environment is ready to run the network:
```
python main.py --cfg cfg/eval_bird.yml --gpu 0
```
By running the following command in the terminal, I got the following error:
```
RuntimeError: CUDA out of memory. Tried to allocate 160.00 MiB (GPU 0; 3.81 GiB total
capacity; 3.08 GiB already allocated; 55.38 MiB free; 9.78 MiB cached)
```

After some investigation, it was figured out that AttnGAN can generate three images with different
resolutions, 64x64x3, 128x128x3 and 256x256x3. However, the AttnDCGAN can just generate the
64x64x3 image. Here, AttnGAN is built over the CUB [?] and COCO dataset and AttnDCGAN is
just built over the CUB dataset. It seems that running the AttnGAN network needs at least the 8GB
memory for the GPU. It seems that the AttnGAN is a bigger network compared to the AttnDCGAN.
Thus, due to hardware limitations, I just worked with AttnDCGAN. I should mention that the proposed
networks are successfully installed in two different computers with 4GB 10st series GPU and 6GB 15st
series GPU but none of them can run the AttnGAN.

## Run the code

For running the AttnDCGAN, I run the following command in the terminal:
```
python main.py --cfg cfg/eval_bird_attnDCGAN2.yml --gpu 0
```
This network is just defined in the CUB dataset and it is not trained on the COCO dataset. Thus,
the report is just about generating images from texts describing the different birds.
The ”Config” file of the AttnDCGAN is located in ```/code/cfg/eval_bird_attnDCGAN2.yml``` and it
includes the following parameters:
```
CONFIG_NAME: ’attn2-dcgan’
DATASET_NAME: ’birds’
DATA_DIR: ’../data/birds’
GPU_ID: 3
WORKERS: 1
B_VALIDATION: False
TREE:
BRANCH_NUM: 3
TRAIN:
FLAG: False
NET_G: ’../models/bird_AttnDCGAN2.pth’
B_NET_D: False
BATCH_SIZE: 100
NET_E: ’../DAMSMencoders/bird/text_encoder200.pth’
GAN:
DF_DIM: 64
GF_DIM: 32
Z_DIM: 100
R_NUM: 0
B_DCGAN: True
TEXT:
EMBEDDING_DIM: 256
CAPTIONS_PER_IMAGE: 20
WORDS_NUM: 25
```

To run this network, it is needed to run the main.py which is based on the following:

```
from __future__ import print_function
from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import gc
gc.collect()
torch.cuda.empty_cache()
#os.environ[’PYTORCH_CUDA_ALLOC_CONF’] = ’max_split_size_mb=1’
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), ’./.’)))
sys.path.append(dir_path)
def parse_args():
parser = argparse.ArgumentParser(description=’Train a AttnGAN network’)
parser.add_argument(’--cfg’, dest=’cfg_file’,
help=’optional config file’,
default=’cfg/bird_attn2.yml’, type=str)
parser.add_argument(’--gpu’, dest=’gpu_id’, type=int, default=-1)
parser.add_argument(’--data_dir’, dest=’data_dir’, type=str, default=’’)
parser.add_argument(’--manualSeed’, type=int, help=’manual seed’)
args = parser.parse_args()
return args
def gen_example(wordtoix, algo):
’’’generate images from example sentences’’’
from nltk.tokenize import RegexpTokenizer
filepath = ’%s/example_filenames.txt’ % (cfg.DATA_DIR)
data_dic = {}
with open(filepath, "r") as f:
filenames = f.read().decode(’utf8’).split(’\n’)
for name in filenames:
if len(name) == 0:
continue
filepath = ’%s/%s.txt’ % (cfg.DATA_DIR, name)
with open(filepath, "r") as f:
print(’Load from:’, name)
sentences = f.read().decode(’utf8’).split(’\n’)
# a list of indices for a sentence
captions = []
cap_lens = []
for sent in sentences:
if len(sent) == 0:
continue
sent = sent.replace("\ufffd\ufffd", " ")
tokenizer = RegexpTokenizer(r’\w+’)
tokens = tokenizer.tokenize(sent.lower())
if len(tokens) == 0:
print(’sent’, sent)
continue
rev = []
for t in tokens:
t = t.encode(’ascii’, ’ignore’).decode(’ascii’)
if len(t) > 0 and t in wordtoix:
rev.append(wordtoix[t])
captions.append(rev)
cap_lens.append(len(rev))
max_len = np.max(cap_lens)
sorted_indices = np.argsort(cap_lens)[::-1]
cap_lens = np.asarray(cap_lens)
cap_lens = cap_lens[sorted_indices]
cap_array = np.zeros((len(captions), max_len), dtype=’int64’)
for i in range(len(captions)):
idx = sorted_indices[i]
cap = captions[idx]
c_len = len(cap)
cap_array[i, :c_len] = cap
key = name[(name.rfind(’/’) + 1):]
data_dic[key] = [cap_array, cap_lens, sorted_indices]
algo.gen_example(data_dic)
if __name__ == "__main__":
args = parse_args()
if args.cfg_file is not None:
cfg_from_file(args.cfg_file)
if args.gpu_id != -1:
cfg.GPU_ID = args.gpu_id
else:
cfg.CUDA = False
if args.data_dir != ’’:
cfg.DATA_DIR = args.data_dir
print(’Using config:’)
pprint.pprint(cfg)
if not cfg.TRAIN.FLAG:
args.manualSeed = 100
elif args.manualSeed is None:
args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if cfg.CUDA:
torch.cuda.manual_seed_all(args.manualSeed)
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime(’%Y_%m_%d_%H_%M_%S’)
output_dir = ’../output/%s_%s_%s’ % \
(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
split_dir, bshuffle = ’train’, True
if not cfg.TRAIN.FLAG:
# bshuffle = False
split_dir = ’test’
# Get data loader
imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
image_transform = transforms.Compose([
transforms.Scale(int(imsize * 76 / 64)),
transforms.RandomCrop(imsize),
transforms.RandomHorizontalFlip()])
dataset = TextDataset(cfg.DATA_DIR, split_dir,
base_size=cfg.TREE.BASE_SIZE,
transform=image_transform)
assert dataset
dataloader = torch.utils.data.DataLoader(
dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
# Define models and go to train/evaluate
algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
start_t = time.time()
if cfg.TRAIN.FLAG:
algo.train()
else:
’’’generate images from pre-extracted embeddings’’’
if cfg.B_VALIDATION:
algo.sampling(split_dir) # generate images for the whole valid dataset
else:
gen_example(dataset.wordtoix, algo) # generate images for customized captions
end_t = time.time()
print(’Total time for training:’, end_t - start_t)
```

For inserting new text descriptions to the network, there is a file whose name is ”example captions.txt”
in ~/AttnGAN/data/birds. The location of this file is defined
in the ”main.py” and it contains the list of text descriptions that should be inserted into the network.
After running the code, the generated images will be located in a folder with the name ”example captions”
in the ~/AttnGAN/models/bird_AttnDCGAN2 repository.
