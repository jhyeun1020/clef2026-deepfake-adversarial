"""
2단계 추가 공격: nonface 1단계 결과에서 MesoNet 제외 3개+ 탐지기 실패한 25장
1단계 delta 복원 → SRM/f3net/sdxl_det/effort 가중치로 15 iteration 추가 공격
GPU 0 사용
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys, torch, cv2, io, random, importlib, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pandas as pd
import mediapipe as mp
from pathlib import Path
from PIL import Image
from torch.amp import autocast
import json

DEVICE = torch.device('cuda:0')
BASE_DIR = Path('/home/student/GEN_AI_GTG')
WEIGHTS_DIR = BASE_DIR / 'weights'
HF_TOKEN = os.environ.get("HF_TOKEN")  # set via environment variable

with open(BASE_DIR / 'images/files/generation_constraints.json') as f:
    constraints_all = json.load(f)

# DiffJPEG
sys.path.insert(0, str(BASE_DIR / 'DiffJPEG'))
from DiffJPEG import DiffJPEG
diff_jpeg_80 = DiffJPEG(height=256, width=256, differentiable=True, quality=80).to(DEVICE)
diff_jpeg_50 = DiffJPEG(height=256, width=256, differentiable=True, quality=50).to(DEVICE)
diff_jpeg_65 = DiffJPEG(height=256, width=256, differentiable=True, quality=65).to(DEVICE)

def input_diversity_extended(x, p=0.5):
    if random.random() > p: return x
    transform = random.choice(['resize', 'blur', 'noise', 'color'])
    if transform == 'resize':
        rnd = random.randint(224, 256)
        x = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
        pad_h = 256 - rnd; pad_w = 256 - rnd
        pt = random.randint(0, pad_h); pl = random.randint(0, pad_w)
        x = F.pad(x, (pl, pad_w - pl, pt, pad_h - pt), value=0)
    elif transform == 'blur':
        k = random.choice([3, 5]); sigma = random.uniform(0.5, 1.5)
        x = TF.gaussian_blur(x, kernel_size=k, sigma=sigma)
    elif transform == 'noise':
        x = (x + torch.randn_like(x) * random.uniform(0.01, 0.03)).clamp(0, 1)
    elif transform == 'color':
        x = (x * random.uniform(0.9, 1.1)).clamp(0, 1)
    return x

def defense_transform(x, p=0.3):
    if random.random() > p: return x
    transform = random.choice(['jpeg_recompress', 'median_approx', 'strong_blur', 'resize_down_up', 'bit_reduce', 'face_crop'])
    if transform == 'jpeg_recompress':
        jpeg_fn = random.choice([diff_jpeg_50, diff_jpeg_65]); x = jpeg_fn(x)
    elif transform == 'median_approx':
        k = random.choice([3, 5]); x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k//2)
    elif transform == 'strong_blur':
        k = random.choice([5, 7]); sigma = random.uniform(1.5, 3.0); x = TF.gaussian_blur(x, kernel_size=k, sigma=sigma)
    elif transform == 'resize_down_up':
        small = random.choice([128, 160, 192]); x = F.interpolate(x, size=(small, small), mode='bilinear', align_corners=False); x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
    elif transform == 'bit_reduce':
        levels = random.choice([16, 32, 64]); x_q = torch.round(x * levels) / levels; x = x + (x_q - x).detach()
    elif transform == 'face_crop':
        margin = random.randint(30, 70); x = x[:, :, margin:256-margin, margin:256-margin]; x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
    return x.clamp(0, 1)

# ── 모델 로드 ──
sys.path.insert(0, str(BASE_DIR / 'deepfakebench_src'))
import transformers.modeling_utils as _tmu; _tmu.check_torch_load_is_safe = lambda: None
import transformers.utils.import_utils as _tiu; _tiu.check_torch_load_is_safe = lambda: None
from transformers import AutoModelForImageClassification, AutoImageProcessor, CLIPModel, AutoProcessor

print("모델 로드 중...", flush=True)

class _HFWrapper(nn.Module):
    def __init__(s,m,p):super().__init__();s.model=m;s.register_buffer('mean',torch.tensor(p.image_mean).view(1,3,1,1));s.register_buffer('std',torch.tensor(p.image_std).view(1,3,1,1));h=p.size.get('height',p.size.get('shortest_edge',224));s.size=(h,p.size.get('width',h))
    def forward(s,x):return s.model((F.interpolate(x,s.size,mode='bilinear',align_corners=False)-s.mean)/s.std)
class _DFBWrapper(nn.Module):
    def __init__(s,m):super().__init__();s.model=m;s.register_buffer('mean',torch.tensor([.5,.5,.5]).view(1,3,1,1));s.register_buffer('std',torch.tensor([.5,.5,.5]).view(1,3,1,1))
    def forward(s,x):return s.model((F.interpolate(x,(256,256),mode='bilinear',align_corners=False)-s.mean)/s.std)
class _SingleLogitWrapper(nn.Module):
    def __init__(s,m,sz,mean,std,use_half=False):super().__init__();s.model=m;s.input_size=sz;s.use_half=use_half;s.register_buffer('mean',torch.tensor(mean).view(1,3,1,1));s.register_buffer('std',torch.tensor(std).view(1,3,1,1))
    def forward(s,x):
        inp=(F.interpolate(x,s.input_size,mode='bilinear',align_corners=False)-s.mean)/s.std
        if s.use_half:
            with autocast('cuda'):logit=s.model(inp.half()).float()
        else:logit=s.model(inp)
        logit=logit.reshape(-1,1);return type('O',(),{'logits':torch.cat([logit,-logit],dim=-1)})()
class _NPRWrapper(nn.Module):
    def __init__(s,m):super().__init__();s.model=m;s.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1));s.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(s,x):x=F.interpolate(x,(256,256),mode='bilinear',align_corners=False);x_up=F.interpolate(F.interpolate(x,scale_factor=0.5,mode='nearest'),scale_factor=2.0,mode='nearest');inp=((x-x_up)*2./3.-s.mean)/s.std;logit=s.model(inp).reshape(-1,1);return type('O',(),{'logits':torch.cat([logit,-logit],dim=-1)})()
class _EnsembleWrapper(nn.Module):
    def __init__(s,e,sp):super().__init__();s.ensemble=e;s.register_buffer('sig_mean',torch.tensor(sp.image_processor.image_mean).view(1,3,1,1));s.register_buffer('sig_std',torch.tensor(sp.image_processor.image_std).view(1,3,1,1));s.sig_size=(sp.image_processor.size.get('height',384),sp.image_processor.size.get('width',384));s.register_buffer('dino_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1));s.register_buffer('dino_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(s,x):
        si=(F.interpolate(x,s.sig_size,mode='bilinear',align_corners=False)-s.sig_mean)/s.sig_std;d=(F.interpolate(x,(392,392),mode='bilinear',align_corners=False)-s.dino_mean)/s.dino_std
        with autocast('cuda'):logits,_,_=s.ensemble(si,d)
        return type('O',(),{'logits':logits.float()})()

from huggingface_hub import hf_hub_download
import importlib.util
REPO_ID_SD='Bombek1/ai-image-detector-siglip-dinov2';model_py=hf_hub_download(repo_id=REPO_ID_SD,filename='model.py');spec=importlib.util.spec_from_file_location('ai_det',model_py);ai_mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(ai_mod)
wt=hf_hub_download(repo_id=REPO_ID_SD,filename='pytorch_model.pt');ckpt=torch.load(wt,map_location='cpu',weights_only=False);cfg=ckpt.get('config',{})
ens_m=ai_mod.create_model_with_lora(siglip_model_name=cfg.get('siglip_model','google/siglip2-so400m-patch14-384'),dinov2_model_name=cfg.get('dinov2_model','vit_large_patch14_dinov2.lvd142m'),image_size=cfg.get('image_size',392));ens_m.load_state_dict(ckpt['model_state_dict'])
siglip_proc=AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384');ens_wrap=_EnsembleWrapper(ens_m,siglip_proc)
sdxl_wrap=_HFWrapper(AutoModelForImageClassification.from_pretrained('Organika/sdxl-detector',token=HF_TOKEN),AutoImageProcessor.from_pretrained('Organika/sdxl-detector',token=HF_TOKEN))
umm_wrap=_HFWrapper(AutoModelForImageClassification.from_pretrained('umm-maybe/AI-image-detector',token=HF_TOKEN),AutoImageProcessor.from_pretrained('umm-maybe/AI-image-detector',token=HF_TOKEN))
ateeq_wrap=_HFWrapper(AutoModelForImageClassification.from_pretrained('Ateeqq/ai-vs-human-image-detector',token=HF_TOKEN),AutoImageProcessor.from_pretrained('Ateeqq/ai-vs-human-image-detector',token=HF_TOKEN))
haywood_wrap=_HFWrapper(AutoModelForImageClassification.from_pretrained('haywoodsloan/ai-image-detector-dev-deploy',token=HF_TOKEN),AutoImageProcessor.from_pretrained('haywoodsloan/ai-image-detector-dev-deploy',token=HF_TOKEN))
print("  HF models OK", flush=True)

from xception import Xception
class Bottleneck(nn.Module):
    expansion=4
    def __init__(s,ip,p,st=1,ds=None):super().__init__();s.conv1=nn.Conv2d(ip,p,1,bias=False);s.bn1=nn.BatchNorm2d(p);s.conv2=nn.Conv2d(p,p,3,stride=st,padding=1,bias=False);s.bn2=nn.BatchNorm2d(p);s.conv3=nn.Conv2d(p,p*4,1,bias=False);s.bn3=nn.BatchNorm2d(p*4);s.relu=nn.ReLU(inplace=True);s.downsample=ds
    def forward(s,x):i=x;o=s.relu(s.bn1(s.conv1(x)));o=s.relu(s.bn2(s.conv2(o)));o=s.bn3(s.conv3(o));return s.relu(o+(s.downsample(x) if s.downsample else i))
class FreqNet(nn.Module):
    def __init__(s):
        super().__init__();s.inplanes=64;s.weight1=nn.Parameter(torch.randn(64,3,1,1));s.bias1=nn.Parameter(torch.zeros(64));s.weight2=nn.Parameter(torch.randn(64,64,1,1));s.bias2=nn.Parameter(torch.zeros(64));s.weight3=nn.Parameter(torch.randn(256,256,1,1));s.bias3=nn.Parameter(torch.zeros(256));s.weight4=nn.Parameter(torch.randn(256,256,1,1));s.bias4=nn.Parameter(torch.zeros(256))
        s.realconv1=nn.Conv2d(64,64,1,bias=False);s.imagconv1=nn.Conv2d(64,64,1,bias=False);s.realconv2=nn.Conv2d(64,64,1,bias=False);s.imagconv2=nn.Conv2d(64,64,1,bias=False);s.realconv3=nn.Conv2d(256,256,1,bias=False);s.imagconv3=nn.Conv2d(256,256,1,bias=False);s.realconv4=nn.Conv2d(256,256,1,bias=False);s.imagconv4=nn.Conv2d(256,256,1,bias=False)
        s.layer1=s._ml(64,3);s.layer2=s._ml(128,4,stride=2);s.avgpool=nn.AdaptiveAvgPool2d((1,1));s.fc1=nn.Linear(512,1)
    def _ml(s,p,b,stride=1):ds=None;(ds:=nn.Sequential(nn.Conv2d(s.inplanes,p*4,1,stride=stride,bias=False),nn.BatchNorm2d(p*4))) if stride!=1 or s.inplanes!=p*4 else None;ly=[Bottleneck(s.inplanes,p,stride,ds)];s.inplanes=p*4;[ly.append(Bottleneck(s.inplanes,p)) for _ in range(1,b)];return nn.Sequential(*ly)
    @staticmethod
    def _hfw(x,sc=4):_,_,H,W=x.shape;f=torch.fft.fft2(x.float(),norm='ortho');fs=torch.fft.fftshift(f);ch,cw=H//2,W//2;sh,sw=H//(2*sc),W//(2*sc);fs[:,:,ch-sh:ch+sh,cw-sw:cw+sw]=0;return F.relu(torch.fft.ifft2(torch.fft.ifftshift(fs),norm='ortho').real)
    @staticmethod
    def _hfc(x,sc=4):B,C,H,W=x.shape;f=torch.fft.fft(x.float().reshape(B,C,-1),dim=1,norm='ortho');fs=torch.fft.fftshift(f,dim=1);cc,sc2=C//2,C//(2*sc);fs[:,cc-sc2:cc+sc2,:]=0;return F.relu(torch.fft.ifft(torch.fft.ifftshift(fs,dim=1),dim=1,norm='ortho').real.reshape(B,C,H,W))
    @staticmethod
    def _fcl(x,rc,ic):f=torch.fft.fft2(x.float(),norm='ortho');r=rc(f.real)-ic(f.imag);i=rc(f.imag)+ic(f.real);return F.relu(torch.fft.ifft2(torch.complex(r,i),norm='ortho').real)
    def forward(s,x):x=s._hfw(x);x=F.relu(F.conv2d(x,s.weight1,s.bias1));x=s._hfc(x);x=s._fcl(x,s.realconv1,s.imagconv1);x=s._hfw(x);x=F.relu(F.conv2d(x,s.weight2,s.bias2,stride=2));x=s._hfc(x);x=s._fcl(x,s.realconv2,s.imagconv2);x=F.max_pool2d(x,3,2,1);x=s.layer1(x);x=s._hfw(x);x=F.relu(F.conv2d(x,s.weight3,s.bias3));x=s._fcl(x,s.realconv3,s.imagconv3);x=s._hfw(x);x=F.relu(F.conv2d(x,s.weight4,s.bias4,stride=2));x=s._fcl(x,s.realconv4,s.imagconv4);x=s.layer2(x);x=s.avgpool(x);return s.fc1(torch.flatten(x,1))
fnet=FreqNet().float();_fs=torch.load(str(WEIGHTS_DIR/'freqnet_v2.pth'),map_location='cpu');fnet.load_state_dict({k.replace('module.',''):v for k,v in(_fs.get('model',_fs)).items()},strict=False)
freq_wrap=_SingleLogitWrapper(fnet,(256,256),[.485,.456,.406],[.229,.224,.225])

import timm,re
class DFC(nn.Module):
    def __init__(s):super().__init__();s.encoder=timm.create_model("tf_efficientnet_b7_ns",pretrained=False,num_classes=0,global_pool="");s.avg_pool=nn.AdaptiveAvgPool2d((1,1));s.fc=nn.Linear(2560,1)
    def forward(s,x):return s.fc(s.avg_pool(s.encoder.forward_features(x)).flatten(1))
dfdc=DFC();_dc=torch.load(str(WEIGHTS_DIR/'dfdc_b7_111.pth'),map_location='cpu');dfdc.load_state_dict({re.sub('^module.','',k):v for k,v in(_dc.get('state_dict',_dc)).items()},strict=False);dfdc.half()
dfdc_wrap=_SingleLogitWrapper(dfdc,(380,380),[.485,.456,.406],[.229,.224,.225],use_half=True)

import mesonet as _mm;importlib.reload(_mm);from mesonet import Meso4
from npr_resnet import resnet50 as _npr50
xc=Xception({'mode':'original','num_classes':2,'inc':3,'dropout':False});xc.load_state_dict({k.replace('backbone.',''):v for k,v in torch.load(str(WEIGHTS_DIR/'xception_best.pth'),map_location='cpu',weights_only=False).items() if k.startswith('backbone.')})
class _XW(nn.Module):
    def __init__(s,b):super().__init__();s.b=b
    def forward(s,x):l,_=s.b(x);return type('O',(),{'logits':l})()
xception_wrap=_DFBWrapper(_XW(xc))
me=Meso4({'num_classes':2,'inc':3});me.load_state_dict({k.replace('backbone.',''):v for k,v in torch.load(str(WEIGHTS_DIR/'meso4_best.pth'),map_location='cpu',weights_only=False).items() if k.startswith('backbone.')})
class _MW(nn.Module):
    def __init__(s,b):super().__init__();s.b=b
    def forward(s,x):l,_=s.b(x);return type('O',(),{'logits':l})()
meso_wrap=_DFBWrapper(_MW(me))

def _DCT(s):return[[(np.sqrt(1./s) if i==0 else np.sqrt(2./s))*np.cos((j+0.5)*np.pi*i/s) for j in range(s)] for i in range(s)]
def _gf(a,b,s):return[[0. if i+j>b or i+j<a else 1. for j in range(s)] for i in range(s)]
class _Fl(nn.Module):
    def __init__(s,sz,bs,be):super().__init__();s.base=nn.Parameter(torch.tensor(_gf(bs,be,sz)),requires_grad=False);s.learnable=nn.Parameter(torch.randn(sz,sz)*0.1)
    def forward(s,x):return x*(s.base+2.*torch.sigmoid(s.learnable)-1.)
class _FAD(nn.Module):
    def __init__(s,sz):super().__init__();d=torch.tensor(_DCT(sz)).float();s._DCT_all=nn.Parameter(d,requires_grad=False);s._DCT_all_T=nn.Parameter(d.t(),requires_grad=False);s.filters=nn.ModuleList([_Fl(sz,0,sz/2.82),_Fl(sz,sz/2.82,sz/2),_Fl(sz,sz/2,sz*2),_Fl(sz,0,sz*2)])
    def forward(s,x):xf=s._DCT_all@x@s._DCT_all_T;return torch.cat([s._DCT_all_T@f(xf)@s._DCT_all for f in s.filters],1)
class _F3(nn.Module):
    def __init__(s):super().__init__();s.fad=_FAD(256);s.bb=Xception({'mode':'original','num_classes':2,'inc':3,'dropout':0.5});s.bb.conv1=nn.Conv2d(12,32,3,2,0,bias=False)
    def forward(s,x):l,_=s.bb(s.fad(x));return type('O',(),{'logits':l})()
f3=_F3();_f3s=torch.load(str(WEIGHTS_DIR/'f3net_best.pth'),map_location='cpu',weights_only=False);f3.fad.load_state_dict({k.replace('FAD_head.',''):v for k,v in _f3s.items() if k.startswith('FAD_head.')});f3.bb.load_state_dict({k.replace('backbone.',''):v for k,v in _f3s.items() if k.startswith('backbone.')})
f3net_wrap=_DFBWrapper(f3)
npr=_npr50(num_classes=1);npr.load_state_dict({k.replace('module.',''):v for k,v in torch.load(str(WEIGHTS_DIR/'NPR.pth'),map_location='cpu',weights_only=False)['model'].items()})
npr_wrap=_NPRWrapper(npr)

# SRM
def _sk():f1=np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],dtype=float)/4.;f2=np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=float)/12.;f3=np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=float)/2.;return f1,f2,f3
class _SC0(nn.Module):
    def __init__(s,inc=3):super().__init__();s.truc=nn.Hardtanh(-3,3);f1,f2,f3=_sk();s.kernel=nn.Parameter(torch.FloatTensor(np.repeat(np.array([[f1],[f2],[f3]]),inc,axis=1)),requires_grad=False)
    def forward(s,x):return s.truc(F.conv2d(x,s.kernel,stride=1,padding=2))
class _SCS(nn.Module):
    def __init__(s,inc,outc):super().__init__();s.inc=inc;s.truc=nn.Hardtanh(-3,3);f1,f2,f3=_sk();s.kernel=nn.Parameter(torch.FloatTensor(np.repeat(np.array([[f1],[f2],[f3]]),inc,axis=0)),requires_grad=False);s.out_conv=nn.Sequential(nn.Conv2d(3*inc,outc,1,1,0,1,1,bias=False),nn.BatchNorm2d(outc),nn.ReLU(True))
    def forward(s,x):return s.out_conv(s.truc(F.conv2d(x,s.kernel,stride=1,padding=2,groups=s.inc)))
class _CA(nn.Module):
    def __init__(s,c,r=8):super().__init__();s.avg_pool=nn.AdaptiveAvgPool2d(1);s.max_pool=nn.AdaptiveMaxPool2d(1);s.sharedMLP=nn.Sequential(nn.Conv2d(c,c//r,1,bias=False),nn.ReLU(),nn.Conv2d(c//r,c,1,bias=False));s.sigmoid=nn.Sigmoid()
    def forward(s,x):return s.sigmoid(s.sharedMLP(s.avg_pool(x))+s.sharedMLP(s.max_pool(x)))
class _SA(nn.Module):
    def __init__(s,ks=7):super().__init__();s.conv=nn.Conv2d(2,1,ks,padding=3,bias=False);s.sig=nn.Sigmoid()
    def forward(s,x):return s.sig(s.conv(torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True)[0]],1)))
class _SPA(nn.Module):
    def __init__(s,inc):super().__init__();s.srm=_SC0();s.conv=nn.Sequential(nn.Conv2d(inc,32,3,2,0,bias=False),nn.BatchNorm2d(32),nn.ReLU(True),nn.Conv2d(32,64,3,bias=False),nn.BatchNorm2d(64),nn.ReLU(True));s.pa=_SA()
    def forward(s,x):return s.pa(s.conv(s.srm(x)))
class _DCMA(nn.Module):
    def __init__(s,d,sz=16,r=8):super().__init__();rr=d//r;s.key_conv1=nn.Conv2d(d,rr,1);s.key_conv2=nn.Conv2d(d,rr,1);s.key_conv_share=nn.Conv2d(rr,rr,1);s.linear1=nn.Linear(sz*sz,sz*sz);s.linear2=nn.Linear(sz*sz,sz*sz);s.value_conv1=nn.Conv2d(d,d,1);s.gamma1=nn.Parameter(torch.zeros(1));s.value_conv2=nn.Conv2d(d,d,1);s.gamma2=nn.Parameter(torch.zeros(1));s.softmax=nn.Softmax(dim=-1)
    def forward(s,x,y):B,C,H,W=x.size();n=H*W;k1=s.key_conv_share(s.key_conv1(x)).view(B,-1,n).permute(0,2,1);k2=s.key_conv_share(s.key_conv2(y)).view(B,-1,n);e=torch.bmm(k1,k2);a1=s.softmax(s.linear1(e));a2=s.softmax(s.linear2(e.permute(0,2,1)));return s.gamma1*torch.bmm(s.value_conv2(y).view(B,-1,n),a1.permute(0,2,1)).view(B,C,H,W)+x,s.gamma2*torch.bmm(s.value_conv1(x).view(B,-1,n),a2.permute(0,2,1)).view(B,C,H,W)+y
class _FM(nn.Module):
    def __init__(s,ic=4096,oc=2048):super().__init__();s.convblk=nn.Sequential(nn.Conv2d(ic,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU());s.ca=_CA(oc,r=16)
    def forward(s,x,y):f=s.convblk(torch.cat((x,y),1));return f*s.ca(f)
class _SRM(nn.Module):
    def __init__(s):
        super().__init__();cfg={'mode':'original','num_classes':2,'inc':3,'dropout':False};s.backbone_rgb=Xception(cfg);s.backbone_srm=Xception(cfg);s.blur=nn.Module();s.blur.kernel_size=7;k=torch.arange(7,dtype=torch.float32)-3;gk=torch.exp(-k**2/1.28);gk=(gk.unsqueeze(0)*gk.unsqueeze(1));gk=gk/gk.sum();s.blur.register_buffer('weight',gk.view(1,1,7,7).repeat(3,1,1,1));s.srm_conv0=_SC0(inc=3);s.srm_conv1=_SCS(32,32);s.srm_conv2=_SCS(64,64);s.relu=nn.ReLU(inplace=True);s.srm_sa=_SPA(3);s.srm_sa_post=nn.Sequential(nn.BatchNorm2d(64),nn.ReLU(True));s.dual_cma0=_DCMA(728);s.dual_cma1=_DCMA(728);s.fusion=_FM()
    def forward(s,x):
        srm=s.srm_conv0(x);xr=s.backbone_rgb.fea_part1_0(x);yr=s.relu(s.backbone_srm.fea_part1_0(srm)+s.srm_conv1(xr));xr=s.backbone_rgb.fea_part1_1(xr);yr=s.relu(s.backbone_srm.fea_part1_1(yr)+s.srm_conv2(xr));xr=xr*s.srm_sa(srm)+xr;xr=s.srm_sa_post(xr);xr=s.backbone_rgb.fea_part2(xr);yr=s.backbone_srm.fea_part2(yr);xr,yr=s.dual_cma0(xr,yr);xr=s.backbone_rgb.fea_part3(xr);yr=s.backbone_srm.fea_part3(yr);xr,yr=s.dual_cma1(xr,yr);xr=s.backbone_rgb.fea_part4(xr);yr=s.backbone_srm.fea_part4(yr);xr=s.backbone_rgb.fea_part5(xr);yr=s.backbone_srm.fea_part5(yr);return s.backbone_rgb.classifier(s.fusion(xr,yr))
srm_m=_SRM();srm_m.load_state_dict(torch.load(str(WEIGHTS_DIR/'srm_best.pth'),map_location='cpu',weights_only=False))
class _SRMW(nn.Module):
    def __init__(s,m):super().__init__();s.m=m
    def forward(s,x):return type('O',(),{'logits':s.m(x)})()
srm_wrap=_DFBWrapper(_SRMW(srm_m))

# Effort
class SVDResidualLinear(nn.Module):
    def __init__(s,inf,outf,r,bias=True,iw=None):super().__init__();s.weight_main=nn.Parameter(torch.zeros(outf,inf),requires_grad=False);s.bias=nn.Parameter(torch.zeros(outf)) if bias else None;s.S_residual=nn.Parameter(torch.zeros(1));s.U_residual=nn.Parameter(torch.zeros(outf,1));s.V_residual=nn.Parameter(torch.zeros(1,inf))
    def forward(s,x):return F.linear(x,s.weight_main+s.U_residual@torch.diag(s.S_residual)@s.V_residual,s.bias)
def _asvd(model,r):
    for name,module in model.named_children():
        if 'self_attn' in name:
            for sn,sm in module.named_modules():
                if isinstance(sm,nn.Linear):
                    parent=module;parts=sn.split('.')
                    for p in parts[:-1]:parent=getattr(parent,p)
                    new=SVDResidualLinear(sm.in_features,sm.out_features,r,sm.bias is not None,sm.weight.data)
                    if sm.bias is not None:new.bias.data.copy_(sm.bias.data)
                    setattr(parent,parts[-1],new)
        else:_asvd(module,r)
    return model
_clip=CLIPModel.from_pretrained('openai/clip-vit-large-patch14');_clip.vision_model=_asvd(_clip.vision_model,r=1023)
eff_bb=_clip.vision_model;eff_hd=nn.Linear(1024,2);_es=torch.load(str(WEIGHTS_DIR/'effort_clip_L14_trainOn_sdv14.pth'),map_location='cpu',weights_only=False);_esd={k.replace('module.',''):v for k,v in _es.items()};eff_bb.load_state_dict({k.replace('backbone.',''):v for k,v in _esd.items() if k.startswith('backbone.')});eff_hd.load_state_dict({k.replace('head.',''):v for k,v in _esd.items() if k.startswith('head.')});del _clip,_es,_esd
class _EW(nn.Module):
    def __init__(s,b,h):super().__init__();s.b=b;s.h=h;s.register_buffer('mean',torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1));s.register_buffer('std',torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1))
    def forward(s,x):x=F.interpolate(x,(224,224),mode='bilinear',align_corners=False);return type('O',(),{'logits':s.h(s.b((x-s.mean)/s.std).pooler_output)})()
effort_wrap=_EW(eff_bb,eff_hd)
print("  ALL models OK", flush=True)

_identity=lambda x:x
ALL_MODELS={
    'siglip_dino':(ens_wrap,_identity,0),'sdxl_det':(sdxl_wrap,_identity,0),
    'umm_maybe':(umm_wrap,_identity,0),'ateeq':(ateeq_wrap,_identity,0),
    'haywood':(haywood_wrap,_identity,0),'freqnet':(freq_wrap,_identity,0),
    'f3net':(f3net_wrap,_identity,1),'npr':(npr_wrap,_identity,0),
    'srm':(srm_wrap,_identity,1),'xception':(xception_wrap,_identity,1),
    'dfdc_b7':(dfdc_wrap,_identity,0),'meso4':(meso_wrap,_identity,1),
    'effort':(effort_wrap,_identity,1),
}
for n,(m,_,_) in ALL_MODELS.items():m.to(DEVICE).eval()

def _e2d(logits):return logits.unsqueeze(0) if logits.dim()==1 else logits

# 2단계 가중치 (SRM, f3net, sdxl_det, effort + dfdc_b7 악화 방지)
STAGE2_WEIGHTS = {
    'srm':0.25, 'f3net':0.20, 'sdxl_det':0.10, 'effort':0.10, 'dfdc_b7':0.05,
    'siglip_dino':0.0, 'umm_maybe':0.0, 'ateeq':0.0, 'haywood':0.0,
    'freqnet':0.0, 'npr':0.0, 'xception':0.0, 'meso4':0.0,
}

def get_scores_batch_s2(x_adv, models_dict):
    scores = {}
    with torch.no_grad():
        x_jpeg = diff_jpeg_80(x_adv)
        for name, (model, pre_fn, ai_idx) in models_dict.items():
            if STAGE2_WEIGHTS.get(name, 0.0) <= 0: continue
            logits = _e2d(model(pre_fn(x_jpeg)).logits)
            if logits.shape[-1] == 1: scores[name] = torch.sigmoid(logits).reshape(-1)
            else: scores[name] = torch.softmax(logits, dim=-1)[:, ai_idx]
    return scores

def attack_stage2(x_batch, delta_init, keys_batch, iterations=15, epsilon=2/255, alpha=0.5/255, momentum_decay=1.0):
    x = x_batch.clone().to(DEVICE); B = x.shape[0]
    delta = delta_init.clone().to(DEVICE)
    momentum = torch.zeros_like(x)

    active_models = {n: v for n, v in ALL_MODELS.items() if STAGE2_WEIGHTS.get(n, 0.0) > 0}
    model_names = list(active_models.keys())
    base_weights = torch.tensor([STAGE2_WEIGHTS[n] for n in model_names], device=DEVICE)
    per_image_weights = base_weights.unsqueeze(0).expand(B, -1).clone()

    for step in range(iterations):
        # adaptive weight (매 5 step)
        if step % 5 == 0 and step > 0:
            with torch.no_grad():
                cs = get_scores_batch_s2(torch.clamp(x + delta, 0, 1), active_models)
                for mi, name in enumerate(model_names):
                    if name not in cs: continue
                    s = cs[name]
                    scale = torch.ones(B, device=DEVICE)
                    scale = torch.where(s < 0.3, torch.tensor(0.5, device=DEVICE), scale)
                    scale = torch.where(s > 0.5, torch.tensor(2.0, device=DEVICE), scale)
                    per_image_weights[:, mi] = base_weights[mi] * scale

        grad_accum = torch.zeros_like(x)
        for mi, name in enumerate(model_names):
            model, pre_fn, ai_idx = active_models[name]
            w = per_image_weights[:, mi].view(B, 1, 1, 1)
            dp = delta.detach().requires_grad_(True)
            x_adv = torch.clamp(x + dp, 0, 1)
            x_jpeg = diff_jpeg_80(x_adv)
            x_def = defense_transform(x_jpeg, p=0.3)
            x_div = input_diversity_extended(x_def, p=0.5)
            logits = _e2d(model(pre_fn(x_div)).logits)
            if logits.shape[-1] == 1:
                loss = (w.squeeze() * logits.reshape(-1)).mean()
            else:
                hi = 1 - ai_idx if logits.shape[-1] == 2 else ai_idx
                loss = (w.squeeze() * (logits[:, ai_idx] - logits[:, hi])).mean()
            loss.backward()
            grad_accum += dp.grad.detach()

        with torch.no_grad():
            current_alpha = alpha * (1.2 if step < iterations * 0.6 else 0.7)
            gn = grad_accum / (grad_accum.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
            momentum = momentum_decay * momentum + gn
            delta -= current_alpha * momentum.sign()
            delta.clamp_(-epsilon, epsilon)

    with torch.no_grad(): return torch.clamp(x + delta, 0, 1)

@torch.no_grad()
def score_all(x_01):
    x=x_01.to(DEVICE);scores={}
    for name,(model,pre_fn,ai_idx) in ALL_MODELS.items():
        logits=_e2d(model(pre_fn(x)).logits)
        if logits.shape[-1]==1:scores[name]=torch.sigmoid(logits).reshape(-1).cpu()
        else:scores[name]=torch.softmax(logits,dim=-1)[:,ai_idx].cpu()
    return scores

# ── 25장 선별 ──
df_s1 = pd.read_csv(BASE_DIR / 'outputs/attack_1002_all_nonface/results_nonface.csv')
ORIG_DIR = BASE_DIR / 'outputs/final_1002'
RAW_DIR = BASE_DIR / 'outputs/attack_1002_all_nonface/raw_png'

exclude_dets = ['meso4']
all_dets_check = [d for d in ['siglip_dino','sdxl_det','umm_maybe','ateeq','haywood','freqnet',
    'f3net','npr','srm','xception','dfdc_b7','effort',
    'aide_progan','aide_genimage','aide_sd14','distildire','recce','ucf'] if f'{d}_full' in df_s1.columns]

target_keys = []
for _, row in df_s1.iterrows():
    fails = [d for d in all_dets_check if row.get(f'{d}_full', 0) >= 0.5]
    if len(fails) >= 3:
        target_keys.append(row['key'])

print(f"\n2단계 공격 대상: {len(target_keys)}장", flush=True)

# 원본 + 1단계 raw 로드, delta 복원
orig_tensors = []
delta_inits = []
for key in target_keys:
    orig = Image.open(ORIG_DIR / f'{key}.png').convert('RGB')
    if orig.size != (256,256): orig = orig.resize((256,256), Image.LANCZOS)
    orig_t = T.ToTensor()(orig)
    raw = Image.open(RAW_DIR / f'{key}.png').convert('RGB')
    raw_t = T.ToTensor()(raw)
    delta = raw_t - orig_t  # 1단계 delta 복원
    orig_tensors.append(orig_t)
    delta_inits.append(delta)

orig_batch = torch.stack(orig_tensors)
delta_batch = torch.stack(delta_inits)
print(f"delta 복원 완료. L_inf: {delta_batch.abs().max():.5f} ({delta_batch.abs().max()*255:.1f}/255)", flush=True)

# ── 1단계 점수 (before) ──
print("\n1단계 점수 기록...", flush=True)
s1_scores = {}
for i, key in enumerate(target_keys):
    row = df_s1[df_s1['key'] == key].iloc[0]
    s1_scores[key] = {det: row[f'{det}_full'] for det in ALL_MODELS if f'{det}_full' in row.index}

# ── 2단계 공격 ──
print("\n2단계 공격 시작...", flush=True)
OUT = BASE_DIR / 'outputs/attack_stage2_25_v2'
(OUT / 'raw_png').mkdir(parents=True, exist_ok=True)
(OUT / 'jpeg_q80').mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5
all_adv = []
t0 = time.time()
for start in range(0, len(target_keys), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(target_keys))
    x_adv = attack_stage2(orig_batch[start:end], delta_batch[start:end], target_keys[start:end])
    all_adv.append(x_adv.cpu())
    for i in range(end - start):
        T.ToPILImage()(x_adv[i].cpu()).save(OUT / 'raw_png' / f'{target_keys[start+i]}.png')
    print(f"  [{end}/{len(target_keys)}]", flush=True)

elapsed = time.time() - t0
adv_tensor = torch.cat(all_adv, 0)
print(f"2단계 공격 완료: {elapsed:.0f}초", flush=True)

# JPEG Q=80
final_pils = []
for i, key in enumerate(target_keys):
    pil = T.ToPILImage()(adv_tensor[i])
    buf = io.BytesIO(); pil.save(buf, format='JPEG', quality=80); buf.seek(0)
    fp = Image.open(buf).convert('RGB')
    fp.save(OUT / 'jpeg_q80' / f'{key}.png')
    final_pils.append(fp)
final_tensors = torch.stack([T.ToTensor()(p) for p in final_pils])

# ── 2단계 점수 (after) ──
print("\n2단계 점수 측정...", flush=True)
s2_scores = {}
for i, key in enumerate(target_keys):
    sa = score_all(final_tensors[i:i+1])
    s2_scores[key] = {det: round(float(sa[det][0]), 4) for det in sa}

# ── 결과 비교 ──
N = len(target_keys)
det_names = list(ALL_MODELS.keys())

print(f"\n{'='*120}", flush=True)
print(f"2단계 공격 결과 비교 ({N}장)", flush=True)
print(f"{'='*120}", flush=True)
print(f"{'탐지기':15s} | {'1단계 avg':>10s} | {'2단계 avg':>10s} | {'delta':>8s} | {'1단계 회피':>10s} | {'2단계 회피':>10s}", flush=True)
print('-' * 80, flush=True)

for det in det_names:
    s1_vals = [s1_scores[k].get(det, 0.5) for k in target_keys]
    s2_vals = [s2_scores[k][det] for k in target_keys]
    s1_avg = np.mean(s1_vals); s2_avg = np.mean(s2_vals)
    s1_ev = sum(1 for v in s1_vals if v < 0.5)
    s2_ev = sum(1 for v in s2_vals if v < 0.5)
    d = s2_avg - s1_avg
    mark = ' +++' if d < -0.05 else (' ---' if d > 0.05 else '')
    print(f"  {det:15s} | {s1_avg:10.4f} | {s2_avg:10.4f} | {d:+8.4f}{mark} | {s1_ev:4d}/{N} | {s2_ev:4d}/{N}", flush=True)

print(f"\n  L_inf: {(adv_tensor - orig_batch).abs().max().item():.5f} ({(adv_tensor - orig_batch).abs().max().item()*255:.1f}/255)", flush=True)

# 개별 키 상세 (SRM, f3net)
print(f"\n{'='*100}", flush=True)
print(f"SRM/f3net 개별 변화:", flush=True)
print(f"{'key':25s} | {'srm_s1':>7s} | {'srm_s2':>7s} | {'f3_s1':>7s} | {'f3_s2':>7s} | {'개선':>4s}", flush=True)
print('-' * 80, flush=True)
for key in target_keys:
    srm1 = s1_scores[key].get('srm', 0.5); srm2 = s2_scores[key]['srm']
    f3_1 = s1_scores[key].get('f3net', 0.5); f3_2 = s2_scores[key]['f3net']
    improved = (srm2 < srm1 - 0.05) or (f3_2 < f3_1 - 0.05)
    print(f"  {key:25s} | {srm1:7.4f} | {srm2:7.4f} | {f3_1:7.4f} | {f3_2:7.4f} | {'O' if improved else 'X'}", flush=True)

# CSV 저장
rows = []
for key in target_keys:
    row = {'key': key}
    for det in det_names:
        row[f'{det}_stage1'] = round(s1_scores[key].get(det, 0.5), 4)
        row[f'{det}_stage2'] = round(s2_scores[key][det], 4)
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / 'results_stage2.csv', index=False)

print(f"\n결과: {OUT}/results_stage2.csv", flush=True)
print("완료 ✓", flush=True)
