from dgfa import DGFA
from omegaconf import OmegaConf
import pretrainedmodels
import os
import torch
from torchvision import transforms as T
from loader import ImageNet,Normalize,TfNormalize
import torch.nn as nn
import copy
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as T
from torchvision.models import maxvit_t,vit_b_16
from functools import partial
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    
setup_seed(2023)


def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/DGFA.yaml")
args = parser.parse_args()
config = OmegaConf.load(args.config)
PREFIX = os.path.basename(args.config)[:-5]
METHOD = eval(config.attack.method)
BATCH_SIZE = config.data.batch_size
EPS = config.attack.eps / 255
NUM_IMAGES = config.data.num_images
STEPS = config.attack.steps
MODEL = config.model.name
if config.attack.method == "DGFA":
    RHO = config.attack.rho
    DELTA = config.attack.delta
    N = config.attack.N
ALPHA = EPS / STEPS
OUTPUT_DIR = "output/"+PREFIX+"/"

def partial_save_image(names):
    return partial(save_image,names=names)
transforms = T.Compose(
    [T.Resize(224 if MODEL in ['maxvit_t','vit_b_16'] else 299
            ), T.ToTensor()]
)

dataset = ImageNet("dataset/images","dataset/images.csv",transforms=transforms,num_images=NUM_IMAGES)


all_images = torch.load("all_images.pt") if not "vit" in MODEL else torch.load("all_images_224.pt")
all_labels = torch.load("all_labels.pt")
all_images_ID = torch.load("all_images_ID.pt")

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

if MODEL in ['maxvit_t','vit_b_16']:
    model = eval(MODEL)(pretrained=True).eval().to(device)
    model = torch.nn.Sequential(Normalize(mean, std),model).eval().to(device)
else:
    model = eval("pretrainedmodels."+MODEL)
    model = torch.nn.Sequential(Normalize(mean, std),model(num_classes=1000, pretrained='imagenet').eval().to(device)).eval().to(device)

attack = METHOD(copy.deepcopy(model),eps=EPS,alpha=ALPHA,steps=STEPS,N=N,delta=DELTA,rho=RHO)
pbar = tqdm(total=1000)
for i in range(1000):
    images = all_images[i].unsqueeze(0).to(device)
    gt = all_labels[i].unsqueeze(0).to(device)
    images_id = all_images_ID[i]
    
    if os.path.exists(OUTPUT_DIR++images_id[0]):
        pbar.update(1)

    adv_images = attack(images,gt,partial_save_image(images_id),output_dir=OUTPUT_DIR)
    pbar.update(1)


pbar.close()
