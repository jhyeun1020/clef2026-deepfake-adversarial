"""
1002장 전체 No-blur PGD + DI + MI 공격 + JPEG Q=80 + 전체 탐지기 검증 + id_sim/lm_dist
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

with open(BASE_DIR / 'images/files/generation_constraints.json') as f:
    constraints_all = json.load(f)

# ── DiffJPEG ──
sys.path.insert(0, str(BASE_DIR / 'DiffJPEG'))
from DiffJPEG import DiffJPEG
diff_jpeg_80 = DiffJPEG(height=256, width=256, differentiable=True, quality=80).to(DEVICE)

# ── Input Diversity ──
def input_diversity(x, low=224, high=256, p=0.5):
    if random.random() > p: return x
    rnd = random.randint(low, high)
    x_r = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    pad_h = 256 - rnd; pad_w = 256 - rnd
    pt = random.randint(0, pad_h); pl = random.randint(0, pad_w)
    return F.pad(x_r, (pl, pad_w - pl, pt, pad_h - pt), value=0)

# ── 모델 로드 ──
sys.path.insert(0, str(BASE_DIR / 'deepfakebench_src'))

import transformers.modeling_utils as _tmu; _tmu.check_torch_load_is_safe = lambda: None
import transformers.utils.import_utils as _tiu; _tiu.check_torch_load_is_safe = lambda: None
from transformers import AutoModelForImageClassification, AutoImageProcessor, CLIPModel, AutoProcessor

HF_TOKEN = os.environ.get("HF_TOKEN")  # set via environment variable
WEIGHTS_DIR = BASE_DIR / 'weights'

print("모델 로드 중...")

# ── Wrapper 클래스 ──
class _HFWrapper(nn.Module):
    def __init__(self, model, processor):
        super().__init__(); self.model = model
        self.register_buffer('mean', torch.tensor(processor.image_mean).view(1,3,1,1))
        self.register_buffer('std', torch.tensor(processor.image_std).view(1,3,1,1))
        h = processor.size.get('height', processor.size.get('shortest_edge', 224))
        self.size = (h, processor.size.get('width', h))
    def forward(self, x):
        return self.model((F.interpolate(x, self.size, mode='bilinear', align_corners=False) - self.mean) / self.std)

class _DFBWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
        self.register_buffer('mean', torch.tensor([0.5,0.5,0.5]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.5,0.5,0.5]).view(1,3,1,1))
    def forward(self, x):
        return self.model((F.interpolate(x, (256,256), mode='bilinear', align_corners=False) - self.mean) / self.std)

class _SingleLogitWrapper(nn.Module):
    def __init__(self, model, input_size, mean, std, use_half=False):
        super().__init__(); self.model = model; self.input_size = input_size; self.use_half = use_half
        self.register_buffer('mean', torch.tensor(mean).view(1,3,1,1))
        self.register_buffer('std', torch.tensor(std).view(1,3,1,1))
    def forward(self, x):
        inp = (F.interpolate(x, self.input_size, mode='bilinear', align_corners=False) - self.mean) / self.std
        if self.use_half:
            with autocast('cuda'): logit = self.model(inp.half()).float()
        else: logit = self.model(inp)
        logit = logit.reshape(-1, 1)
        return type('Out', (), {'logits': torch.cat([logit, -logit], dim=-1)})()

class _NPRWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self, x):
        x = F.interpolate(x, (256,256), mode='bilinear', align_corners=False)
        x_up = F.interpolate(F.interpolate(x, scale_factor=0.5, mode='nearest'), scale_factor=2.0, mode='nearest')
        inp = ((x - x_up) * 2.0/3.0 - self.mean) / self.std
        logit = self.model(inp).reshape(-1, 1)
        return type('Out', (), {'logits': torch.cat([logit, -logit], dim=-1)})()

class _EnsembleWrapper(nn.Module):
    def __init__(self, ensemble, siglip_proc):
        super().__init__(); self.ensemble = ensemble
        self.register_buffer('sig_mean', torch.tensor(siglip_proc.image_processor.image_mean).view(1,3,1,1))
        self.register_buffer('sig_std', torch.tensor(siglip_proc.image_processor.image_std).view(1,3,1,1))
        self.sig_size = (siglip_proc.image_processor.size.get('height',384), siglip_proc.image_processor.size.get('width',384))
        self.register_buffer('dino_mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('dino_std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self, x):
        s = (F.interpolate(x, self.sig_size, mode='bilinear', align_corners=False) - self.sig_mean) / self.sig_std
        d = (F.interpolate(x, (392,392), mode='bilinear', align_corners=False) - self.dino_mean) / self.dino_std
        with autocast('cuda'):
            logits, _, _ = self.ensemble(s, d)
        return type('Out', (), {'logits': logits.float()})()

# ── 모델 인스턴스 ──
# SigLIP2+DINOv2
from huggingface_hub import hf_hub_download
import importlib.util
REPO_ID_SD = 'Bombek1/ai-image-detector-siglip-dinov2'
model_py = hf_hub_download(repo_id=REPO_ID_SD, filename='model.py')
spec = importlib.util.spec_from_file_location('ai_det', model_py)
ai_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ai_mod)
wt = hf_hub_download(repo_id=REPO_ID_SD, filename='pytorch_model.pt')
ckpt = torch.load(wt, map_location='cpu', weights_only=False)
cfg = ckpt.get('config', {})
ens_m = ai_mod.create_model_with_lora(
    siglip_model_name=cfg.get('siglip_model','google/siglip2-so400m-patch14-384'),
    dinov2_model_name=cfg.get('dinov2_model','vit_large_patch14_dinov2.lvd142m'),
    image_size=cfg.get('image_size',392))
ens_m.load_state_dict(ckpt['model_state_dict'])
siglip_proc = AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
ens_wrap = _EnsembleWrapper(ens_m, siglip_proc)
print("  SigLIP2+DINOv2 ✓")

# HF models
sdxl_wrap = _HFWrapper(AutoModelForImageClassification.from_pretrained('Organika/sdxl-detector',token=HF_TOKEN),
                       AutoImageProcessor.from_pretrained('Organika/sdxl-detector',token=HF_TOKEN))
umm_wrap = _HFWrapper(AutoModelForImageClassification.from_pretrained('umm-maybe/AI-image-detector',token=HF_TOKEN),
                      AutoImageProcessor.from_pretrained('umm-maybe/AI-image-detector',token=HF_TOKEN))
ateeq_wrap = _HFWrapper(AutoModelForImageClassification.from_pretrained('Ateeqq/ai-vs-human-image-detector',token=HF_TOKEN),
                        AutoImageProcessor.from_pretrained('Ateeqq/ai-vs-human-image-detector',token=HF_TOKEN))
haywood_wrap = _HFWrapper(AutoModelForImageClassification.from_pretrained('haywoodsloan/ai-image-detector-dev-deploy',token=HF_TOKEN),
                          AutoImageProcessor.from_pretrained('haywoodsloan/ai-image-detector-dev-deploy',token=HF_TOKEN))
print("  HF models ✓")

# FreqNet
from xception import Xception
class Bottleneck(nn.Module):
    expansion=4
    def __init__(s,ip,p,st=1,ds=None):
        super().__init__();s.conv1=nn.Conv2d(ip,p,1,bias=False);s.bn1=nn.BatchNorm2d(p);s.conv2=nn.Conv2d(p,p,3,stride=st,padding=1,bias=False);s.bn2=nn.BatchNorm2d(p);s.conv3=nn.Conv2d(p,p*4,1,bias=False);s.bn3=nn.BatchNorm2d(p*4);s.relu=nn.ReLU(inplace=True);s.downsample=ds
    def forward(s,x):
        i=x;o=s.relu(s.bn1(s.conv1(x)));o=s.relu(s.bn2(s.conv2(o)));o=s.bn3(s.conv3(o))
        if s.downsample:i=s.downsample(x)
        return s.relu(o+i)
class FreqNet(nn.Module):
    def __init__(s):
        super().__init__();s.inplanes=64
        s.weight1=nn.Parameter(torch.randn(64,3,1,1));s.bias1=nn.Parameter(torch.zeros(64));s.weight2=nn.Parameter(torch.randn(64,64,1,1));s.bias2=nn.Parameter(torch.zeros(64))
        s.weight3=nn.Parameter(torch.randn(256,256,1,1));s.bias3=nn.Parameter(torch.zeros(256));s.weight4=nn.Parameter(torch.randn(256,256,1,1));s.bias4=nn.Parameter(torch.zeros(256))
        s.realconv1=nn.Conv2d(64,64,1,bias=False);s.imagconv1=nn.Conv2d(64,64,1,bias=False);s.realconv2=nn.Conv2d(64,64,1,bias=False);s.imagconv2=nn.Conv2d(64,64,1,bias=False)
        s.realconv3=nn.Conv2d(256,256,1,bias=False);s.imagconv3=nn.Conv2d(256,256,1,bias=False);s.realconv4=nn.Conv2d(256,256,1,bias=False);s.imagconv4=nn.Conv2d(256,256,1,bias=False)
        s.layer1=s._ml(64,3);s.layer2=s._ml(128,4,stride=2);s.avgpool=nn.AdaptiveAvgPool2d((1,1));s.fc1=nn.Linear(512,1)
    def _ml(s,p,b,stride=1):
        ds=None
        if stride!=1 or s.inplanes!=p*4:ds=nn.Sequential(nn.Conv2d(s.inplanes,p*4,1,stride=stride,bias=False),nn.BatchNorm2d(p*4))
        ly=[Bottleneck(s.inplanes,p,stride,ds)];s.inplanes=p*4
        for _ in range(1,b):ly.append(Bottleneck(s.inplanes,p))
        return nn.Sequential(*ly)
    @staticmethod
    def _hfw(x,sc=4):
        _,_,H,W=x.shape;f=torch.fft.fft2(x.float(),norm='ortho');fs=torch.fft.fftshift(f);ch,cw=H//2,W//2;sh,sw=H//(2*sc),W//(2*sc);fs[:,:,ch-sh:ch+sh,cw-sw:cw+sw]=0
        return F.relu(torch.fft.ifft2(torch.fft.ifftshift(fs),norm='ortho').real)
    @staticmethod
    def _hfc(x,sc=4):
        B,C,H,W=x.shape;f=torch.fft.fft(x.float().reshape(B,C,-1),dim=1,norm='ortho');fs=torch.fft.fftshift(f,dim=1);cc,sc2=C//2,C//(2*sc);fs[:,cc-sc2:cc+sc2,:]=0
        return F.relu(torch.fft.ifft(torch.fft.ifftshift(fs,dim=1),dim=1,norm='ortho').real.reshape(B,C,H,W))
    @staticmethod
    def _fcl(x,rc,ic):
        f=torch.fft.fft2(x.float(),norm='ortho');r=rc(f.real)-ic(f.imag);i=rc(f.imag)+ic(f.real)
        return F.relu(torch.fft.ifft2(torch.complex(r,i),norm='ortho').real)
    def forward(s,x):
        x=s._hfw(x);x=F.relu(F.conv2d(x,s.weight1,s.bias1));x=s._hfc(x);x=s._fcl(x,s.realconv1,s.imagconv1)
        x=s._hfw(x);x=F.relu(F.conv2d(x,s.weight2,s.bias2,stride=2));x=s._hfc(x);x=s._fcl(x,s.realconv2,s.imagconv2)
        x=F.max_pool2d(x,3,2,1);x=s.layer1(x);x=s._hfw(x);x=F.relu(F.conv2d(x,s.weight3,s.bias3));x=s._fcl(x,s.realconv3,s.imagconv3)
        x=s._hfw(x);x=F.relu(F.conv2d(x,s.weight4,s.bias4,stride=2));x=s._fcl(x,s.realconv4,s.imagconv4);x=s.layer2(x);x=s.avgpool(x);return s.fc1(torch.flatten(x,1))
fnet=FreqNet().float();_fs=torch.load(str(WEIGHTS_DIR/'freqnet_v2.pth'),map_location='cpu');
if isinstance(_fs,dict) and 'model' in _fs:_fs=_fs['model']
fnet.load_state_dict({k.replace('module.',''):v for k,v in _fs.items()},strict=False)
freq_wrap=_SingleLogitWrapper(fnet,(256,256),[0.485,0.456,0.406],[0.229,0.224,0.225])
print("  FreqNet ✓")

# DFDC
import timm,re
class DFC(nn.Module):
    def __init__(s):super().__init__();s.encoder=timm.create_model("tf_efficientnet_b7_ns",pretrained=False,num_classes=0,global_pool="");s.avg_pool=nn.AdaptiveAvgPool2d((1,1));s.fc=nn.Linear(2560,1)
    def forward(s,x):return s.fc(s.avg_pool(s.encoder.forward_features(x)).flatten(1))
dfdc=DFC();_dc=torch.load(str(WEIGHTS_DIR/'dfdc_b7_111.pth'),map_location='cpu');dfdc.load_state_dict({re.sub('^module.','',k):v for k,v in(_dc.get('state_dict',_dc)).items()},strict=False);dfdc.half()
dfdc_wrap=_SingleLogitWrapper(dfdc,(380,380),[0.485,0.456,0.406],[0.229,0.224,0.225],use_half=True)
print("  DFDC ✓")

# XceptionNet, MesoNet, F3Net, NPR, SRM
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

# F3Net
def _DCT(s):return[[(np.sqrt(1./s) if i==0 else np.sqrt(2./s))*np.cos((j+0.5)*np.pi*i/s) for j in range(s)] for i in range(s)]
def _gf(a,b,s):return[[0. if i+j>b or i+j<a else 1. for j in range(s)] for i in range(s)]
class _Fl(nn.Module):
    def __init__(s,sz,bs,be):super().__init__();s.base=nn.Parameter(torch.tensor(_gf(bs,be,sz)),requires_grad=False);s.learnable=nn.Parameter(torch.randn(sz,sz)*0.1)
    def forward(s,x):return x*(s.base+2.*torch.sigmoid(s.learnable)-1.)
class _FAD(nn.Module):
    def __init__(s,sz):
        super().__init__();d=torch.tensor(_DCT(sz)).float();s._DCT_all=nn.Parameter(d,requires_grad=False);s._DCT_all_T=nn.Parameter(d.t(),requires_grad=False)
        s.filters=nn.ModuleList([_Fl(sz,0,sz/2.82),_Fl(sz,sz/2.82,sz/2),_Fl(sz,sz/2,sz*2),_Fl(sz,0,sz*2)])
    def forward(s,x):xf=s._DCT_all@x@s._DCT_all_T;return torch.cat([s._DCT_all_T@f(xf)@s._DCT_all for f in s.filters],1)
class _F3(nn.Module):
    def __init__(s):super().__init__();s.fad=_FAD(256);s.bb=Xception({'mode':'original','num_classes':2,'inc':3,'dropout':0.5});s.bb.conv1=nn.Conv2d(12,32,3,2,0,bias=False)
    def forward(s,x):l,_=s.bb(s.fad(x));return type('O',(),{'logits':l})()
f3=_F3();_f3s=torch.load(str(WEIGHTS_DIR/'f3net_best.pth'),map_location='cpu',weights_only=False)
f3.fad.load_state_dict({k.replace('FAD_head.',''):v for k,v in _f3s.items() if k.startswith('FAD_head.')})
f3.bb.load_state_dict({k.replace('backbone.',''):v for k,v in _f3s.items() if k.startswith('backbone.')})
f3net_wrap=_DFBWrapper(f3)

npr=_npr50(num_classes=1);npr.load_state_dict({k.replace('module.',''):v for k,v in torch.load(str(WEIGHTS_DIR/'NPR.pth'),map_location='cpu',weights_only=False)['model'].items()})
npr_wrap=_NPRWrapper(npr)

# SRM
def _sk():
    f1=np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],dtype=float)/4.
    f2=np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=float)/12.
    f3=np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=float)/2.
    return f1,f2,f3
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
    def __init__(s,d,sz=16,r=8):
        super().__init__();rr=d//r;s.key_conv1=nn.Conv2d(d,rr,1);s.key_conv2=nn.Conv2d(d,rr,1);s.key_conv_share=nn.Conv2d(rr,rr,1)
        s.linear1=nn.Linear(sz*sz,sz*sz);s.linear2=nn.Linear(sz*sz,sz*sz);s.value_conv1=nn.Conv2d(d,d,1);s.gamma1=nn.Parameter(torch.zeros(1))
        s.value_conv2=nn.Conv2d(d,d,1);s.gamma2=nn.Parameter(torch.zeros(1));s.softmax=nn.Softmax(dim=-1)
    def forward(s,x,y):
        B,C,H,W=x.size();n=H*W;k1=s.key_conv_share(s.key_conv1(x)).view(B,-1,n).permute(0,2,1);k2=s.key_conv_share(s.key_conv2(y)).view(B,-1,n)
        e=torch.bmm(k1,k2);a1=s.softmax(s.linear1(e));a2=s.softmax(s.linear2(e.permute(0,2,1)))
        return s.gamma1*torch.bmm(s.value_conv2(y).view(B,-1,n),a1.permute(0,2,1)).view(B,C,H,W)+x,s.gamma2*torch.bmm(s.value_conv1(x).view(B,-1,n),a2.permute(0,2,1)).view(B,C,H,W)+y
class _FM(nn.Module):
    def __init__(s,ic=4096,oc=2048):super().__init__();s.convblk=nn.Sequential(nn.Conv2d(ic,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU());s.ca=_CA(oc,r=16)
    def forward(s,x,y):f=s.convblk(torch.cat((x,y),1));return f*s.ca(f)
class _SRM(nn.Module):
    def __init__(s):
        super().__init__();cfg={'mode':'original','num_classes':2,'inc':3,'dropout':False}
        s.backbone_rgb=Xception(cfg);s.backbone_srm=Xception(cfg);s.blur=nn.Module();s.blur.kernel_size=7
        k=torch.arange(7,dtype=torch.float32)-3;gk=torch.exp(-k**2/1.28);gk=(gk.unsqueeze(0)*gk.unsqueeze(1));gk=gk/gk.sum()
        s.blur.register_buffer('weight',gk.view(1,1,7,7).repeat(3,1,1,1))
        s.srm_conv0=_SC0(inc=3);s.srm_conv1=_SCS(32,32);s.srm_conv2=_SCS(64,64);s.relu=nn.ReLU(inplace=True)
        s.srm_sa=_SPA(3);s.srm_sa_post=nn.Sequential(nn.BatchNorm2d(64),nn.ReLU(True));s.dual_cma0=_DCMA(728);s.dual_cma1=_DCMA(728);s.fusion=_FM()
    def forward(s,x):
        srm=s.srm_conv0(x);xr=s.backbone_rgb.fea_part1_0(x);yr=s.relu(s.backbone_srm.fea_part1_0(srm)+s.srm_conv1(xr))
        xr=s.backbone_rgb.fea_part1_1(xr);yr=s.relu(s.backbone_srm.fea_part1_1(yr)+s.srm_conv2(xr))
        xr=xr*s.srm_sa(srm)+xr;xr=s.srm_sa_post(xr);xr=s.backbone_rgb.fea_part2(xr);yr=s.backbone_srm.fea_part2(yr)
        xr,yr=s.dual_cma0(xr,yr);xr=s.backbone_rgb.fea_part3(xr);yr=s.backbone_srm.fea_part3(yr)
        xr,yr=s.dual_cma1(xr,yr);xr=s.backbone_rgb.fea_part4(xr);yr=s.backbone_srm.fea_part4(yr)
        xr=s.backbone_rgb.fea_part5(xr);yr=s.backbone_srm.fea_part5(yr)
        return s.backbone_rgb.classifier(s.fusion(xr,yr))
srm_m=_SRM();srm_m.load_state_dict(torch.load(str(WEIGHTS_DIR/'srm_best.pth'),map_location='cpu',weights_only=False))
class _SRMW(nn.Module):
    def __init__(s,m):super().__init__();s.m=m
    def forward(s,x):return type('O',(),{'logits':s.m(x)})()
srm_wrap=_DFBWrapper(_SRMW(srm_m))

# Effort
class SVDResidualLinear(nn.Module):
    def __init__(s,inf,outf,r,bias=True,iw=None):
        super().__init__();s.weight_main=nn.Parameter(torch.zeros(outf,inf),requires_grad=False)
        if bias:s.bias=nn.Parameter(torch.zeros(outf))
        else:s.register_parameter('bias',None)
        s.S_residual=nn.Parameter(torch.zeros(1));s.U_residual=nn.Parameter(torch.zeros(outf,1));s.V_residual=nn.Parameter(torch.zeros(1,inf))
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
eff_bb=_clip.vision_model;eff_hd=nn.Linear(1024,2)
_es=torch.load(str(WEIGHTS_DIR/'effort_clip_L14_trainOn_sdv14.pth'),map_location='cpu',weights_only=False)
_esd={k.replace('module.',''):v for k,v in _es.items()}
eff_bb.load_state_dict({k.replace('backbone.',''):v for k,v in _esd.items() if k.startswith('backbone.')})
eff_hd.load_state_dict({k.replace('head.',''):v for k,v in _esd.items() if k.startswith('head.')})
del _clip,_es,_esd
class _EW(nn.Module):
    def __init__(s,b,h):super().__init__();s.b=b;s.h=h;s.register_buffer('mean',torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1));s.register_buffer('std',torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1))
    def forward(s,x):x=F.interpolate(x,(224,224),mode='bilinear',align_corners=False);return type('O',(),{'logits':s.h(s.b((x-s.mean)/s.std).pooler_output)})()
effort_wrap=_EW(eff_bb,eff_hd)
print("  Effort + XceptionNet + MesoNet + F3Net + NPR + SRM ✓")

# ── 전부 GPU ──
_identity = lambda x: x
ALL_MODELS = {
    'siglip_dino':(ens_wrap,_identity,0),'sdxl_det':(sdxl_wrap,_identity,0),
    'umm_maybe':(umm_wrap,_identity,0),'ateeq':(ateeq_wrap,_identity,0),
    'haywood':(haywood_wrap,_identity,0),'freqnet':(freq_wrap,_identity,0),
    'f3net':(f3net_wrap,_identity,1),'npr':(npr_wrap,_identity,0),
    'srm':(srm_wrap,_identity,1),'xception':(xception_wrap,_identity,1),
    'dfdc_b7':(dfdc_wrap,_identity,0),'meso4':(meso_wrap,_identity,1),
    'effort':(effort_wrap,_identity,1),
}
for n,(m,_,_) in ALL_MODELS.items(): m.to(DEVICE).eval()
print(f"\n모든 모델 GPU: {len(ALL_MODELS)}개")

WEIGHTS = {
    'srm':0.08,'meso4':0.05,'freqnet':0.00,'npr':0.00,
    'umm_maybe':0.02,'haywood':0.02,'dfdc_b7':0.05,
    'siglip_dino':0.08,'ateeq':0.08,
    'sdxl_det':0.18,'f3net':0.11,'xception':0.08,'effort':0.16,
}

def _e2d(logits):
    return logits.unsqueeze(0) if logits.dim()==1 else logits

# ── 공격 함수 ──
def attack_noblur(x_batch,iterations=25,epsilon=2/255,alpha=0.5/255,momentum_decay=1.0):
    x=x_batch.clone().to(DEVICE);delta=torch.zeros_like(x);momentum=torch.zeros_like(x)
    for step in range(iterations):
        grad_accum=torch.zeros_like(x);loss_dict={}
        for name,(model,pre_fn,ai_idx) in ALL_MODELS.items():
            w=WEIGHTS.get(name,0.0)
            if w<=0:continue
            dp=delta.detach().requires_grad_(True);x_adv=torch.clamp(x+dp,0,1);x_jpeg=diff_jpeg_80(x_adv)
            x_div=input_diversity(x_jpeg,low=224,high=256,p=0.5)
            logits=_e2d(model(pre_fn(x_div)).logits)
            if logits.shape[-1]==1:loss=w*logits.mean()
            else:
                hi=1-ai_idx if logits.shape[-1]==2 else ai_idx
                loss=w*(logits[:,ai_idx].mean()-logits[:,hi].mean())
            loss.backward();grad_accum+=dp.grad.detach()
        with torch.no_grad():
            gn=grad_accum/(grad_accum.abs().mean(dim=[1,2,3],keepdim=True)+1e-8)
            momentum=momentum_decay*momentum+gn;delta-=alpha*momentum.sign();delta.clamp_(-epsilon,epsilon)
    with torch.no_grad():return torch.clamp(x+delta,0,1)

@torch.no_grad()
def score_all(x_01):
    x=x_01.to(DEVICE);scores={}
    for name,(model,pre_fn,ai_idx) in ALL_MODELS.items():
        logits=_e2d(model(pre_fn(x)).logits)
        if logits.shape[-1]==1:scores[name]=torch.sigmoid(logits).reshape(-1).cpu()
        else:scores[name]=torch.softmax(logits,dim=-1)[:,ai_idx].cpu()
    return scores

# ── InsightFace + MediaPipe ──
from insightface.app import FaceAnalysis
face_app=FaceAnalysis(name='buffalo_s',root=str(BASE_DIR/'images/models'),providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1)
video_ref_embs=np.load(str(BASE_DIR/'models/video_ref_embs.npy'),allow_pickle=True).item()

def extract_lm(pil):
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True) as fm:
        res=fm.process(np.array(pil.convert('RGB')))
    if not res.multi_face_landmarks:return None
    return [(l.x,l.y) for l in res.multi_face_landmarks[0].landmark]

def euc_dist(g,r):
    if g is None or len(g)!=len(r):return 0.5
    return float(np.mean(np.sqrt(np.sum((np.array(g)-np.array(r))**2,axis=1))))

# ── 1002장 로드 (512→256 INTER_AREA) ──
print("\n이미지 로드 중...")
IMAGES_DIR = BASE_DIR / 'app/input/res/Images'
all_keys = sorted(constraints_all.keys())
all_ids = [k.rsplit('_GT_',1)[0] for k in all_keys]

all_tensors = []
all_pils = []
for key in all_keys:
    img = Image.open(IMAGES_DIR / f'{key}.png').convert('RGB')
    bgr = cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (256,256), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    all_pils.append(pil)
    all_tensors.append(T.ToTensor()(pil))
all_tensors = torch.stack(all_tensors)
print(f"로드 완료: {all_tensors.shape}")

# ── 공격 ──
OUT = BASE_DIR / 'outputs/attack_1002'
(OUT / 'raw_png').mkdir(parents=True, exist_ok=True)
(OUT / 'jpeg_q80').mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
all_adv = []
t0 = time.time()

for start in range(0, len(all_keys), BATCH_SIZE):
    end = min(start+BATCH_SIZE, len(all_keys))
    batch = all_tensors[start:end]
    x_adv = attack_noblur(batch)
    all_adv.append(x_adv.cpu())
    for i in range(end-start):
        T.ToPILImage()(x_adv[i].cpu()).save(OUT / 'raw_png' / f'{all_keys[start+i]}.png')
    if (end) % 100 == 0 or end == len(all_keys):
        elapsed = time.time()-t0
        eta = (elapsed/(end))*(len(all_keys)-end)
        print(f"  [{end}/{len(all_keys)}] {elapsed:.0f}초, ETA {eta/60:.0f}분", flush=True)

elapsed_attack = time.time()-t0
adv_tensor = torch.cat(all_adv, 0)
print(f"\n공격 완료: {elapsed_attack:.0f}초 ({elapsed_attack/len(all_keys):.1f}초/장)")

# JPEG Q=80
print("JPEG Q=80 적용...")
final_pils = []
for i, key in enumerate(all_keys):
    pil = T.ToPILImage()(adv_tensor[i])
    buf = io.BytesIO(); pil.save(buf, format='JPEG', quality=80); buf.seek(0)
    fp = Image.open(buf).convert('RGB')
    fp.save(OUT / 'jpeg_q80' / f'{key}.png')
    final_pils.append(fp)
final_tensors = torch.stack([T.ToTensor()(p) for p in final_pils])

# ── 전체 평가 ──
print("\n전체 평가 중...")
rows = []
for i, key in enumerate(all_keys):
    id_str = all_ids[i]
    sa = score_all(final_tensors[i:i+1])

    # id_sim
    ref_emb = video_ref_embs.get(id_str)
    img_bgr = cv2.cvtColor(np.array(final_pils[i]), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bgr)
    if faces:
        emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)
        id_sim = float(np.dot(emb, ref_emb)) if ref_emb is not None else -1
        detected = True
    else:
        id_sim = -1; detected = False

    # lm_dist
    facemesh_lm = constraints_all[key]['mediapipe_landmarks_facemesh']
    gen_lm = extract_lm(final_pils[i])
    lm_dist = euc_dist(gen_lm, facemesh_lm)

    row = {'key': key, 'id': id_str, 'id_sim': round(id_sim,4), 'lm_dist': round(lm_dist,6), 'face_detected': detected}
    for det in sa:
        row[f'{det}'] = round(float(sa[det][0]), 4)
    rows.append(row)

    if (i+1) % 200 == 0:
        print(f"  평가 [{i+1}/{len(all_keys)}]", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'results_1002.csv', index=False)

# ── 요약 ──
print(f"\n{'='*70}")
print(f"1002장 공격 + 평가 결과 (No-blur PGD + DI + MI, JPEG Q=80)")
print(f"{'='*70}")

det_cols = [c for c in df.columns if c not in ['key','id','id_sim','lm_dist','face_detected']]

print(f"\n── 탐지기 회피율 ──")
print(f"{'탐지기':15s} {'평균':>8s} {'회피':>8s}")
print('-'*35)
for det in det_cols:
    avg = df[det].mean()
    evade = (df[det] < 0.5).sum()
    print(f"  {det:15s} {avg:8.4f} {evade:>5d}/1002")

print(f"\n  id_sim 평균: {df['id_sim'].mean():.4f}")
print(f"  lm_dist 평균: {df['lm_dist'].mean():.6f}")
print(f"  얼굴 미감지: {(~df['face_detected']).sum()}장")

# 공통 실패 ID
print(f"\n── 여러 탐지기에서 공통 실패하는 ID ──")
fail_counts = {}
for _, row in df.iterrows():
    fails = sum(1 for det in det_cols if row[det] >= 0.5)
    if fails >= 2:
        fail_counts[row['key']] = {'fails': fails, 'id_sim': row['id_sim'], 'lm_dist': row['lm_dist']}
        for det in det_cols:
            if row[det] >= 0.5:
                fail_counts[row['key']][det] = round(row[det], 3)

if fail_counts:
    print(f"2개 이상 탐지기 미통과: {len(fail_counts)}장")
    for key in sorted(fail_counts, key=lambda k: -fail_counts[k]['fails']):
        info = fail_counts[key]
        failed_dets = [d for d in det_cols if d in info]
        print(f"  {key:25s} fails={info['fails']} id_sim={info['id_sim']:.3f} | {', '.join(f'{d}={info[d]:.3f}' for d in failed_dets)}")
else:
    print("  없음 ✓")

print(f"\n공격 소요: {elapsed_attack:.0f}초")
print(f"결과: {OUT}/results_1002.csv")
print("완료 ✓")
