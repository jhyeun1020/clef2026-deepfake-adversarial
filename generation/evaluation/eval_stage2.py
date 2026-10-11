"""
2단계 공격 25장: 전이 탐지기 (AIDE + DistilDIRE + RECCE + UCF) 측정
GPU 0 사용
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_TOKEN'] = 'REMOVED_TOKEN_USE_ENV_VAR'

import sys, torch, cv2, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

DEVICE = torch.device('cuda:0')
BASE_DIR = Path('/home/student/GEN_AI_GTG')
WEIGHTS_DIR = BASE_DIR / 'weights'
OUT = BASE_DIR / 'outputs/attack_stage2_25_v2'

df = pd.read_csv(OUT / 'results_stage2.csv')
target_keys = df['key'].tolist()
N = len(target_keys)
print(f"키: {N}장", flush=True)

# 이미지 로드
final_pils = [Image.open(OUT / 'jpeg_q80' / f'{k}.png').convert('RGB') for k in target_keys]

# ── AIDE ──
sys.path.insert(0, str(BASE_DIR / 'AIDE'))
from data.dct import DCT_base_Rec_Module
from models.AIDE import AIDE
dct_module = DCT_base_Rec_Module()
normalize_aide = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def prepare_aide_input(pil_img):
    img_t = T.ToTensor()(pil_img.convert('RGB'))
    img_t = F.interpolate(img_t.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0]
    x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(img_t)
    min1 = normalize_aide(F.interpolate(x_minmin.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0])
    max1 = normalize_aide(F.interpolate(x_maxmax.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0])
    min2 = normalize_aide(F.interpolate(x_minmin1.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0])
    max2 = normalize_aide(F.interpolate(x_maxmax1.unsqueeze(0), (256, 256), mode='bilinear', align_corners=False)[0])
    tokens = normalize_aide(img_t)
    return torch.stack([min1, max1, min2, max2, tokens], dim=0).unsqueeze(0)

AIDE_CKPTS = {
    'aide_progan': str(BASE_DIR / 'AIDE/weights/progan_train.pth'),
    'aide_genimage': str(BASE_DIR / 'AIDE/weights/GenImage_train.pth'),
    'aide_sd14': str(BASE_DIR / 'AIDE/weights/sd14_train.pth'),
}

def score_aide(pils, ckpt_name, ckpt_path):
    print(f"  {ckpt_name}...", flush=True)
    model = AIDE(resnet_path=None, convnext_path='laion2b_s34b_b82k_augreg_soup')
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model' in sd: sd = sd['model']
    model.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}, strict=False)
    model.to(DEVICE).eval()
    scores = []
    with torch.no_grad():
        for pil in pils:
            inp = prepare_aide_input(pil).to(DEVICE)
            logits = model(inp)
            scores.append(torch.softmax(logits, dim=-1)[0, 1].item())
    model.cpu(); del model; torch.cuda.empty_cache()
    return scores

# ── DistilDIRE ──
sys.path.insert(0, str(BASE_DIR / 'DistilDIRE'))
from networks.distill_model import DistilDIRE
from guided_diffusion.compute_dire_eps import dire_get_first_step_noise
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion, model_and_diffusion_defaults, dict_parse)

def score_distildire(pils):
    print(f"  DistilDIRE...", flush=True)
    adm_defaults = model_and_diffusion_defaults()
    adm_args = dict(clip_denoised=True, use_ddim=True, real_step=0, attention_resolutions='32,16,8', class_cond=False, diffusion_steps=1000, image_size=256, learn_sigma=True, model_path=str(BASE_DIR / 'models/256x256-adm.pt'), noise_schedule='linear', num_channels=256, num_head_channels=64, num_res_blocks=2, resblock_updown=True, use_fp16=True, use_scale_shift_norm=True, timestep_respacing='ddim20')
    adm_full = {**adm_defaults, **adm_args}
    adm_model, diffusion = create_model_and_diffusion(**dict_parse(adm_full, model_and_diffusion_defaults().keys()))
    adm_model.load_state_dict(torch.load(adm_full['model_path'], map_location='cpu'))
    adm_model.to(DEVICE).eval()
    if adm_full.get('use_fp16'): adm_model.convert_to_fp16()
    dd_ckpt = torch.load(str(WEIGHTS_DIR / 'distildire.pth'), map_location='cpu')
    distildire = DistilDIRE('cuda').to(DEVICE)
    distildire.load_state_dict({k.replace('module.',''):v for k,v in dd_ckpt.get('model', dd_ckpt).items()})
    distildire.eval()
    dire_transform = T.Compose([T.Resize(256, antialias=True), T.CenterCrop((256, 256))])
    import torchvision.transforms.functional as TFF
    scores = []
    with torch.no_grad():
        for pil in pils:
            img = TFF.to_tensor(pil.convert('RGB'))
            img = dire_transform(img).to(DEVICE) * 2 - 1
            img = img.unsqueeze(0)
            eps = dire_get_first_step_noise(img, adm_model, diffusion, adm_full, 'cuda')
            scores.append(float(distildire(img, eps)['logit'].sigmoid().item()))
    distildire.cpu(); adm_model.cpu(); del distildire, adm_model, diffusion; torch.cuda.empty_cache()
    return scores

# ── RECCE + UCF ──
sys.path.insert(0, str(BASE_DIR / 'deepfakebench_src'))
from xception import Xception as XceptionDFB, SeparableConv2d, Block
from timm.models import xception

class GraphReasoning(nn.Module):
    def __init__(self, va_in, va_out, vb_in, vb_out, vc_in, vc_out, spatial_ratio, drop_rate):
        super().__init__();self.ratio=spatial_ratio;self.va_embedding=nn.Sequential(nn.Conv2d(va_in,va_out,1,bias=False),nn.ReLU(True),nn.Conv2d(va_out,va_out,1,bias=False));self.va_gated_b=nn.Sequential(nn.Conv2d(va_in,va_out,1,bias=False),nn.Sigmoid());self.va_gated_c=nn.Sequential(nn.Conv2d(va_in,va_out,1,bias=False),nn.Sigmoid());self.vb_embedding=nn.Sequential(nn.Linear(vb_in,vb_out,bias=False),nn.ReLU(True),nn.Linear(vb_out,vb_out,bias=False));self.vc_embedding=nn.Sequential(nn.Linear(vc_in,vc_out,bias=False),nn.ReLU(True),nn.Linear(vc_out,vc_out,bias=False));self.unfold_b=nn.Unfold(kernel_size=spatial_ratio[0],stride=spatial_ratio[0]);self.unfold_c=nn.Unfold(kernel_size=spatial_ratio[1],stride=spatial_ratio[1]);self.reweight_ab=nn.Sequential(nn.Linear(va_out+vb_out,1,bias=False),nn.ReLU(True),nn.Softmax(dim=1));self.reweight_ac=nn.Sequential(nn.Linear(va_out+vc_out,1,bias=False),nn.ReLU(True),nn.Softmax(dim=1));self.reproject=nn.Sequential(nn.Conv2d(va_out+vb_out+vc_out,va_in,1,bias=False),nn.ReLU(True),nn.Conv2d(va_in,va_in,1,bias=False),nn.Dropout(drop_rate) if drop_rate else nn.Identity())
    def forward(self, va, vb, vc):
        ea=self.va_embedding(va).reshape([va.shape[0],-1,va.shape[2]*va.shape[3]]);gb=(1-self.va_gated_b(va)).reshape(*ea.shape);gc=(1-self.va_gated_c(va)).reshape(*ea.shape)
        vb=self.unfold_b(vb).reshape([vb.shape[0],vb.shape[1],self.ratio[0]**2,-1]).permute([0,2,3,1]);eb=self.vb_embedding(vb)
        vc=self.unfold_c(vc).reshape([vc.shape[0],vc.shape[1],self.ratio[1]**2,-1]).permute([0,2,3,1]);ec=self.vc_embedding(vc)
        ab,ac=[],[]
        for j in range(ea.shape[-1]):
            a1=torch.stack([ea[:,:,j]]*(self.ratio[0]**2),dim=1);b1=eb[:,:,j,:];w=self.reweight_ab(torch.cat([a1,b1],dim=-1));ab.append(torch.bmm(b1.transpose(1,2),w).squeeze()*gb[:,:,j])
            a2=torch.stack([ea[:,:,j]]*(self.ratio[1]**2),dim=1);c1=ec[:,:,j,:];w=self.reweight_ac(torch.cat([a2,c1],dim=-1));ac.append(torch.bmm(c1.transpose(1,2),w).squeeze()*gc[:,:,j])
        agg=torch.sigmoid(torch.cat([torch.stack(ab,dim=-1),torch.stack(ac,dim=-1),ea],dim=1))
        return self.reproject(agg.reshape(va.shape[0],-1,va.shape[2],va.shape[3]))

class GuidedAttention(nn.Module):
    def __init__(self,depth=728,drop_rate=0.2):super().__init__();self.gated=nn.Sequential(nn.Conv2d(3,3,3,1,1,bias=False),nn.ReLU(True),nn.Conv2d(3,1,1,bias=False),nn.Sigmoid());self.h=nn.Sequential(nn.Conv2d(depth,depth,1,1,bias=False),nn.BatchNorm2d(depth),nn.ReLU(True));self.dropout=nn.Dropout(drop_rate)
    def forward(self,x,px,emb):r=torch.abs(x-px);rx=F.interpolate(r,size=emb.shape[-2:],mode='bilinear',align_corners=True);return self.gated(rx)*self.h(emb)+self.dropout(emb)

class Recce(nn.Module):
    def __init__(self,num_classes=2,drop_rate=0.2):
        super().__init__();self.encoder=xception(pretrained=False);self.global_pool=nn.AdaptiveAvgPool2d((1,1));self.dropout=nn.Dropout(drop_rate);self.fc=nn.Linear(2048,num_classes);self.attention=GuidedAttention(728,drop_rate);self.reasoning=GraphReasoning(728,256,256,256,128,256,[2,4],drop_rate)
        self.decoder1=nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),SeparableConv2d(728,256,3,1,1,bias=False),nn.BatchNorm2d(256),nn.ReLU(True));self.decoder2=Block(256,256,3,1);self.decoder3=nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),SeparableConv2d(256,128,3,1,1,bias=False),nn.BatchNorm2d(128),nn.ReLU(True));self.decoder4=Block(128,128,3,1);self.decoder5=nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),SeparableConv2d(128,64,3,1,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(True));self.decoder6=nn.Sequential(nn.Conv2d(64,3,1,1,bias=False),nn.Tanh())
    def features(self,x):
        o=self.encoder.conv1(x);o=self.encoder.bn1(o);o=self.encoder.act1(o);o=self.encoder.conv2(o);o=self.encoder.bn2(o);o=self.encoder.act2(o);o=self.encoder.block1(o);o=self.encoder.block2(o);o=self.encoder.block3(o);emb=self.encoder.block4(o)
        o=self.dropout(emb);o=self.decoder1(o);d2=self.decoder2(o);o=self.decoder3(d2);d4=self.decoder4(o);o=self.decoder5(d4);pred=self.decoder6(o);rx=F.interpolate(pred,size=x.shape[-2:],mode='bilinear',align_corners=True)
        emb=self.encoder.block5(emb);emb=self.encoder.block6(emb);emb=self.encoder.block7(emb);f=self.reasoning(emb,d2,d4)+emb;emb=self.encoder.block8(f);ia=self.attention(x,rx,emb)
        emb=self.encoder.block9(ia);emb=self.encoder.block10(emb);emb=self.encoder.block11(emb);emb=self.encoder.block12(emb);emb=self.encoder.conv3(emb);emb=self.encoder.bn3(emb);emb=self.encoder.act3(emb);emb=self.encoder.conv4(emb);emb=self.encoder.bn4(emb);emb=self.encoder.act4(emb)
        return self.dropout(self.global_pool(emb).squeeze(2).squeeze(2))
    def forward(self,x):return self.fc(self.features(x))

def r_double_conv(ic,oc):return nn.Sequential(nn.Conv2d(ic,oc,3,padding=1),nn.ReLU(True),nn.Conv2d(oc,oc,3,padding=1),nn.ReLU(True))
class AdaIN(nn.Module):
    def __init__(s,eps=1e-5):super().__init__();s.eps=eps
    def forward(s,x,y):sz=x.size();bs,ch=sz[:2];x_=x.view(bs,ch,-1);y_=y.reshape(bs,ch,-1);xs=(x_.var(-1)+s.eps).sqrt().view(bs,ch,1,1);xm=x_.mean(-1).view(bs,ch,1,1);ys=(y_.var(-1)+s.eps).sqrt().view(bs,ch,1,1);ym=y_.mean(-1).view(bs,ch,1,1);return((x-xm.expand(sz))/xs.expand(sz))*ys.expand(sz)+ym.expand(sz)
class Conditional_UNet(nn.Module):
    def __init__(s):super().__init__();s.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True);s.do=nn.Dropout(p=0.3);s.a3=AdaIN();s.a2=AdaIN();s.a1=AdaIN();s.d3=r_double_conv(512,256);s.d2=r_double_conv(256,128);s.d1=r_double_conv(128,64);s.cl=nn.Conv2d(64,3,1);s.ul=nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True);s.act=nn.Tanh()
    def forward(s,c,x):x=s.a3(x,c);x=s.up(x);x=s.do(x);x=s.d3(x);c=s.up(c);c=s.do(c);c=s.d3(c);x=s.a2(x,c);x=s.up(x);x=s.do(x);x=s.d2(x);c=s.up(c);c=s.do(c);c=s.d2(c);x=s.a1(x,c);x=s.up(x);x=s.do(x);x=s.d1(x);return s.act(s.ul(s.cl(x)))
class Conv2d1x1(nn.Module):
    def __init__(s,i,h,o):super().__init__();s.c=nn.Sequential(nn.Conv2d(i,h,1,1),nn.LeakyReLU(True),nn.Conv2d(h,h,1,1),nn.LeakyReLU(True),nn.Conv2d(h,o,1,1))
    def forward(s,x):return s.c(x)
class Head(nn.Module):
    def __init__(s,i,h,o):super().__init__();s.do=nn.Dropout(0.2);s.pool=nn.AdaptiveAvgPool2d(1);s.mlp=nn.Sequential(nn.Linear(i,h),nn.LeakyReLU(True),nn.Linear(h,o))
    def forward(s,x):bs=x.size()[0];xf=s.pool(x).view(bs,-1);x=s.mlp(xf);x=s.do(x);return x,xf
class UCFModel(nn.Module):
    def __init__(s):super().__init__();cfg={'mode':'adjust_channel','num_classes':2,'inc':3,'dropout':False};s.encoder_f=XceptionDFB(cfg);s.encoder_c=XceptionDFB(cfg);s.encoder_feat_dim=512;h=256;s.con_gan=Conditional_UNet();s.head_spe=Head(h,s.encoder_feat_dim,5);s.head_sha=Head(h,s.encoder_feat_dim,2);s.block_spe=Conv2d1x1(s.encoder_feat_dim,h,h);s.block_sha=Conv2d1x1(s.encoder_feat_dim,h,h)
    def forward(s,x):f=s.encoder_f.features(x);fs=s.block_sha(f);o,_=s.head_sha(fs);return o

norm=T.Normalize([.5,.5,.5],[.5,.5,.5])
@torch.no_grad()
def eval_model(model,pils,label=''):
    scores=[]
    for pil in pils:
        t=T.ToTensor()(pil.convert('RGB'));t=F.interpolate(t.unsqueeze(0),(256,256),mode='bilinear',align_corners=False);t=norm(t[0]).unsqueeze(0).to(DEVICE);logits=model(t);scores.append(round(torch.softmax(logits,dim=-1)[0,1].cpu().item(),4))
    return scores

# ── 실행 ──
print("\n전이 탐지기 측정...", flush=True)
transfer = {}

for cn,cp in AIDE_CKPTS.items():
    transfer[cn] = score_aide(final_pils, cn, cp)

transfer['distildire'] = score_distildire(final_pils)

print("  RECCE...", flush=True)
recce=Recce(2);sd=torch.load(str(WEIGHTS_DIR/'recce_best.pth'),map_location='cpu',weights_only=False);recce.load_state_dict({k.replace('model.',''):v for k,v in sd.items() if k.startswith('model.')},strict=False);recce.to(DEVICE).eval()
transfer['recce'] = eval_model(recce, final_pils)
recce.cpu();del recce;torch.cuda.empty_cache()

print("  UCF...", flush=True)
ucf=UCFModel();ucf.load_state_dict(torch.load(str(WEIGHTS_DIR/'ucf_best.pth'),map_location='cpu',weights_only=False),strict=False);ucf.to(DEVICE).eval()
transfer['ucf'] = eval_model(ucf, final_pils)
ucf.cpu();del ucf;torch.cuda.empty_cache()

# 1단계 전이 탐지기 점수 (nonface CSV에서)
df_s1 = pd.read_csv(BASE_DIR / 'outputs/attack_1002_all_nonface/results_nonface.csv')

# CSV 업데이트 + 결과 출력
transfer_names = ['aide_progan','aide_genimage','aide_sd14','distildire','recce','ucf']
for det in transfer_names:
    df[f'{det}_stage2'] = [round(v,4) for v in transfer[det]]
    # 1단계 값 가져오기
    s1_vals = []
    for key in target_keys:
        row = df_s1[df_s1['key']==key]
        s1_vals.append(float(row[f'{det}_full'].iloc[0]) if f'{det}_full' in row.columns and len(row)>0 else 0.5)
    df[f'{det}_stage1'] = s1_vals

df.to_csv(OUT / 'results_stage2.csv', index=False)

# 전체 표 출력
det_13 = ['siglip_dino','sdxl_det','umm_maybe','ateeq','haywood','freqnet','f3net','npr','srm','xception','dfdc_b7','meso4','effort']
print(f"\n{'='*90}", flush=True)
print(f"2단계 공격 전체 결과 ({N}장) — 13탐지기 + 전이탐지기", flush=True)
print(f"{'='*90}", flush=True)
print(f"{'탐지기':15s} | {'1단계 avg':>10s} | {'2단계 avg':>10s} | {'delta':>8s} | {'1단계 회피':>10s} | {'2단계 회피':>10s}", flush=True)
print('-' * 85, flush=True)

for det in det_13:
    s1 = df[f'{det}_stage1'].values; s2 = df[f'{det}_stage2'].values
    d = np.mean(s2) - np.mean(s1)
    mark = ' +++' if d < -0.05 else (' ---' if d > 0.05 else '')
    print(f"  {det:15s} | {np.mean(s1):10.4f} | {np.mean(s2):10.4f} | {d:+8.4f}{mark} | {(s1<0.5).sum():4d}/{N} | {(s2<0.5).sum():4d}/{N}", flush=True)

for det in transfer_names:
    s1 = df[f'{det}_stage1'].values; s2 = df[f'{det}_stage2'].values
    d = np.mean(s2) - np.mean(s1)
    mark = ' +++' if d < -0.05 else (' ---' if d > 0.05 else '')
    print(f"  {det:15s} | {np.mean(s1):10.4f} | {np.mean(s2):10.4f} | {d:+8.4f}{mark} | {(s1<0.5).sum():4d}/{N} | {(s2<0.5).sum():4d}/{N}", flush=True)

print(f"\n결과: {OUT}/results_stage2.csv", flush=True)
print("완료 ✓", flush=True)
