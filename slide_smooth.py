# #!/usr/bin/env python3
# # ============================================================================
# #   GelSight smooth-texture classifier – “big-GPU” edition
# #   usage:  CUDA_VISIBLE_DEVICES=0 python slide_smooth.py --stage all
# # ============================================================================

# import argparse, glob, os, random, math, json
# from pathlib import Path
# from typing import List, Tuple

# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2
# from PIL import Image

# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchmetrics.classification import MulticlassF1Score
# from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
# from lightly.loss import NTXentLoss
# from lightly.models.utils import deactivate_requires_grad, activate_requires_grad
# try:
#     from lightly.optim import LARS
# except Exception:
#     try: from lightly.optim.lars import LARS
#     except Exception: LARS = None

# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report, confusion_matrix
# from tqdm.auto import tqdm

# # ------------------------- CLI -------------------------------------------
# P = argparse.ArgumentParser()
# P.add_argument("--root", default=".")
# P.add_argument("--stage", choices=["ssl","finetune","all"], default="all")
# P.add_argument("--batch_ssl", type=int, default=256)
# P.add_argument("--batch_sup", type=int, default=512)
# P.add_argument("--epochs_ssl", type=int, default=200)
# P.add_argument("--epochs_sup", type=int, default=60)
# P.add_argument("--folds",      type=int, default=5)
# P.add_argument("--device",     default="cuda")
# args = P.parse_args()

# # ------------------------- misc setup ------------------------------------
# SEED=42
# random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# A.random.seed(SEED)
# DEV=torch.device(args.device if torch.cuda.is_available() else "cpu")
# AMP= DEV.type=="cuda"
# os.makedirs("checkpoints",exist_ok=True)

# IMG_S, IMG_B = 224, 384
# STRIDE_S, STRIDE_B = 112, 192

# # ------------------------- helper: random tile ---------------------------
# def random_tile(img:Image.Image, size:int, stride:int):
#     w,h = img.size
#     x = random.randrange(0, w-size+1, stride)
#     y = random.randrange(0, h-size+1, stride)
#     return img.crop((x,y,x+size,y+size))

# # ------------------------- heavy aug -------------------------------------
# MEAN,STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
# def _albumentations_heavy(sz:int)->A.Compose:
#     return A.Compose([
#         A.RandomRotate90(p=0.5),
#         A.Affine(scale=(0.88,1.12), rotate=(-12,12),
#                  translate_percent=0.07, shear=(-8,8), p=0.7),
#         A.HorizontalFlip(p=0.5),  A.VerticalFlip(p=0.3),
#         A.RandomBrightnessContrast(0.25,0.25,p=0.4),
#         A.HueSaturationValue(8,12,8,p=0.4),
#         A.ImageCompression(60,100,p=0.3),
#         A.GaussNoise(10,40,p=0.4),
#         A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2,
#                        shadow_dimension=5, p=0.25),
#         A.CoarseDropout(max_holes=6, max_height=int(0.12*sz),
#                         max_width=int(0.12*sz), fill_value=None, p=0.4),
#         A.Resize(sz,sz,cv2.INTER_AREA),
#         A.Normalize(MEAN, STD),
#         ToTensorV2()
#     ])

# A_TSM = _albumentations_heavy(IMG_S)
# A_TBG = _albumentations_heavy(IMG_B)
# V_TFM = A.Compose([A.Resize(IMG_S,IMG_S),A.Normalize(MEAN,STD), ToTensorV2()])

# # ------------------------- discovery -------------------------------------
# def discover(root)->Tuple[List[str],List[int],List[float]]:
#     img,lbl,depth = [],[],[]
#     for m in sorted(glob.glob(os.path.join(root,"material_*"))):
#         y=int(m.split("_")[-1])-1
#         for cyc in glob.glob(os.path.join(m,"cycle_*")):
#             meta = Path(cyc)/"meta.json"
#             d_mm = json.load(open(meta))["depth_mm"] if meta.exists() else 1.0
#             for fr in glob.glob(os.path.join(cyc,"frame*.png")):
#                 img.append(fr); lbl.append(y); depth.append(d_mm)
#     return img,lbl,depth

# IMGS,LABELS,DEPTHS = discover(args.root)
# NCLS=len(set(LABELS))
# print(f"✔ {len(IMGS)} frames  |  {NCLS} classes")

# # ------------------------- datasets --------------------------------------
# class SSLTiles(Dataset):
#     def __init__(self, paths):
#         self.paths=paths
#     def __len__(self): return len(self.paths)
#     def __getitem__(self,i):
#         img=Image.open(self.paths[i]).convert("RGB")
#         return (A_TSM(image=np.array(random_tile(img,IMG_S,STRIDE_S)))['image'],
#                 A_TSM(image=np.array(random_tile(img,IMG_S,STRIDE_S)))['image'])

# class SupTiles(Dataset):
#     def __init__(self, paths,lbls,depths,train=True):
#         self.recs=list(zip(paths,lbls,depths)); self.train=train
#     def __len__(self): return len(self.recs)
#     def __getitem__(self,i):
#         p,l,d=self.recs[i]
#         im=Image.open(p).convert("RGB")
#         sm=torch.stack([A_TSM(image=np.array(random_tile(im,IMG_S,STRIDE_S)))['image']
#                         for _ in range(6)])
#         bg=torch.stack([A_TBG(image=np.array(random_tile(im,IMG_B,STRIDE_B)))['image']
#                         for _ in range(4)])
#         depth_code=torch.tensor([math.sin(d/5.0), math.cos(d/5.0)],
#                                 dtype=torch.float32)
#         return sm,bg,l,depth_code

# # -------------------- model ----------------------------------------------
# class LocalGlobal(nn.Module):
#     def __init__(self, dim=2304, heads=4):
#         super().__init__()
#         self.local = nn.MultiheadAttention(dim,heads,batch_first=True)
#         self.glob  = nn.MultiheadAttention(dim,heads,batch_first=True)
#         self.proj  = nn.Linear(dim,dim)
#     def forward(self, x):                 # x: (T,D)
#         loc,_ = self.local(x.unsqueeze(0),x.unsqueeze(0),x.unsqueeze(0))
#         glo,_ = self.glob(loc,loc,loc)
#         return self.proj(glo.mean(1)).squeeze(0)  # (D,)

# class DualPyramid(nn.Module):
#     def __init__(self,nc):
#         super().__init__()
#         self.fe = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
#         for name,p in self.fe.named_parameters():
#             layer=int(name.split('.')[0].lstrip('_blocks') or 0)
#             p.requires_grad = layer>=3            # freeze first 3 stages
#         d = self.fe.classifier[1].in_features
#         self.fe.classifier = nn.Identity()
#         self.attn = LocalGlobal(d)
#         self.head = nn.Sequential(
#             nn.Linear(d*2+2,1024), nn.BatchNorm1d(1024), nn.SiLU(),
#             nn.Dropout(0.4),
#             nn.Linear(1024,512), nn.BatchNorm1d(512), nn.SiLU(),
#             nn.Dropout(0.4),
#             nn.Linear(512,nc))
#     def _agg(self,t):                           # t: (B,T,3,H,W)
#         b,t = t.shape[:2]
#         f = self.fe(t.flatten(0,1))             # (B*T,D)
#         f = f.view(b,-1,f.size(-1))             # (B,T,D)
#         return self.attn(f)                     # (B,D)
#     def forward(self,sm,bg,depth):
#         z=torch.cat([self._agg(sm), self._agg(bg), depth],1)
#         return self.head(z)

# # -------------------- MixUp / CutMix -------------------------------------
# def mixup_cutmix(x,y,alpha=0.4):
#     lam=np.random.beta(alpha,alpha)
#     idx=torch.randperm(x.size(0),device=x.device)
#     if random.random()<0.5:        # MixUp
#         x_mix = lam*x + (1-lam)*x[idx]
#     else:                          # CutMix patch on all tiles
#         b,c,h,w = x.shape
#         cx,cy = np.random.randint(w), np.random.randint(h)
#         cut=int(h*math.sqrt(1-lam))
#         x_mix=x.clone()
#         x_mix[:,:,cy:cy+cut,cx:cx+cut]=x[idx,:,cy:cy+cut,cx:cx+cut]
#     y_mix = lam*y + (1-lam)*y[idx]
#     return x_mix,y_mix

# # -------------------- SSL pre-train --------------------------------------
# def ssl_pretrain(backbone,paths):
#     dl=DataLoader(SSLTiles(paths),batch_size=args.batch_ssl,
#                   shuffle=True,num_workers=8,drop_last=True,pin_memory=True)
#     proj=nn.Sequential(nn.Linear(2304,1024),nn.BatchNorm1d(1024),nn.ReLU(),
#                        nn.Linear(1024,256)).to(DEV)
#     opt_base=optim.SGD(proj.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-6)
#     opt=LARS(opt_base) if LARS else opt_base
#     ntx=NTXentLoss().to(DEV)
#     activate_requires_grad(backbone)
#     for ep in range(1,args.epochs_ssl+1):
#         for v1,v2 in tqdm(dl,desc=f"SSL {ep}/{args.epochs_ssl}"):
#             v1,v2=v1.to(DEV),v2.to(DEV)
#             with torch.autocast(device_type=DEV.type,enabled=AMP):
#                 h1,h2=backbone.fe(v1),backbone.fe(v2)
#             loss=ntx(proj(h1.float()),proj(h2.float()))
#             opt.zero_grad(); loss.backward(); opt.step()
#     activate_requires_grad(backbone)

# # -------------------- supervised train -----------------------------------
# def train_one_fold(net,paths,lbls,depths):
#     tr_ds=SupTiles(paths,lbls,depths,True)
#     va_ds=SupTiles(paths,lbls,depths,False)    # same split object outside
#     dl_tr=DataLoader(tr_ds,batch_size=args.batch_sup,shuffle=True,
#                      num_workers=8,pin_memory=True)
#     dl_va=DataLoader(va_ds,batch_size=args.batch_sup,shuffle=False,
#                      num_workers=4,pin_memory=True)
#     opt=optim.AdamW([p for p in net.parameters() if p.requires_grad],
#                     lr=3e-4,weight_decay=1e-4)
#     sched=optim.lr_scheduler.OneCycleLR(opt,3e-4,len(dl_tr),args.epochs_sup)
#     f1=MulticlassF1Score(NCLS,average='macro').to(DEV)
#     ema=torch.optim.swa_utils.AveragedModel(net)
#     best=0; best_state=None

#     for ep in range(1,args.epochs_sup+1):
#         net.train()
#         for sm,bg,l,code in tqdm(dl_tr,desc=f"Ep{ep}/{args.epochs_sup}"):
#             sm,bg=sm.to(DEV),bg.to(DEV)
#             y=nn.functional.one_hot(l.to(DEV),NCLS).float()
#             sm,bg = mixup_cutmix(sm,bg) if random.random()<0.5 else (sm,bg)
#             opt.zero_grad(set_to_none=True)
#             with torch.autocast(device_type=DEV.type,enabled=AMP):
#                 out=net(sm,bg,code.to(DEV))
#                 loss=(-(y*out.log_softmax(1)).sum(1)).mean()
#             loss.backward(); opt.step(); sched.step(); ema.update_parameters(net)

#         net.eval(); ema.eval(); vp,gt=[],[]
#         with torch.no_grad(), torch.autocast(device_type=DEV.type,enabled=AMP):
#             for sm,bg,l,code in dl_va:
#                 sm,bg=sm.to(DEV),bg.to(DEV)
#                 vp.append(ema(sm,bg,code.to(DEV)).softmax(1))
#                 gt.append(l.to(DEV))
#         score=f1(torch.cat(vp),torch.cat(gt)).item()
#         if score>best: best,best_state=score,ema.module.state_dict().copy()
#         print(f"  val-F1={score:.4f}  best={best:.4f}")
#     return best_state

# # -------------------- K-fold CV ------------------------------------------
# skf=StratifiedKFold(args.folds,shuffle=True,random_state=SEED)
# ALLP,ALLT=[],[]
# for fold,(tr,va) in enumerate(skf.split(IMGS,LABELS),1):
#     print(f"\n── fold {fold}/{args.folds} ──")
#     tr_i=[IMGS[i] for i in tr]; va_i=[IMGS[i] for i in va]
#     tr_l=[LABELS[i] for i in tr]; va_l=[LABELS[i] for i in va]
#     tr_d=[DEPTHS[i] for i in tr]; va_d=[DEPTHS[i] for i in va]

#     mdl=DualPyramid(NCLS).to(DEV)
#     if args.stage in ("ssl","all"):
#         ssl_pretrain(mdl, tr_i)
#     if args.stage in ("finetune","all"):
#         best=train_one_fold(mdl,tr_i,tr_l,tr_d)
#         mdl.load_state_dict(best)
#     torch.save(mdl.state_dict(),f"checkpoints/fold{fold}.pt")

#     # evaluate
#     mdl.eval(); pr,gt=[],[]
#     va_loader=DataLoader(SupTiles(va_i,va_l,va_d,False),
#                          batch_size=args.batch_sup,shuffle=False,num_workers=4)
#     with torch.no_grad(), torch.autocast(device_type=DEV.type,enabled=AMP):
#         for sm,bg,l,code in va_loader:
#             sm,bg=sm.to(DEV),bg.to(DEV)
#             pr.append(mdl(sm,bg,code.to(DEV)).softmax(1)); gt.append(l.to(DEV))
#     ALLP.append(torch.cat(pr)); ALLT.append(torch.cat(gt))

# # -------------------- report ---------------------------------------------
# PRED=torch.cat(ALLP).argmax(1).cpu().numpy()
# TRUE=torch.cat(ALLT).cpu().numpy()
# print("\n=== 5-fold ensemble ===")
# print(classification_report(TRUE,PRED,digits=4))
# print(confusion_matrix(TRUE,PRED))














#!/usr/bin/env python3
# =============================================================================
#  slide_smooth_with3d_and_metric.py 
#
#  - Dual?scale 2D Attention (EfficientNet?B3 backbone + FiLM depth fusion) 
#  - Shallow 3D?CNN over 12?frame volumes for temporal cues 
#  - Auxiliary triplet?margin loss on cycle?level embedding 
#  - Cycle?level voting at inference 
#  - Heavy but realistic augmentations 
# =============================================================================

import torch
print("Torch sees these devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f" Device {i}: {torch.cuda.get_device_name(i)}")
print("Current device index:", torch.cuda.current_device())
print("Selected device:", torch.cuda.get_device_name(torch.cuda.current_device()))

import argparse, glob, os, random, math
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchmetrics.classification import MulticlassF1Score
from torchvision import transforms as T
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

from lightly.loss import NTXentLoss
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.models.utils import deactivate_requires_grad, activate_requires_grad

# Try importing LARS from lightly, else fallback to SGD
try:
    from lightly.optim import LARS
except ImportError:
    try:
        from lightly.optim.lars import LARS
    except ImportError:
        LARS = None  # fallback to plain SGD

# =============================================================================
# 1) CLI + Config
# =============================================================================
parser = argparse.ArgumentParser(
    description="GelSight dual?scale + 3D?CNN pipeline with auxiliary metric loss"
)
parser.add_argument("--root",        default=".", 
                    help="Root folder containing material_* subfolders")
parser.add_argument("--stage",       choices=["ssl", "finetune", "all"], 
                    default="all")
parser.add_argument("--batch_ssl",   type=int, default=256)
parser.add_argument("--batch_sup",   type=int, default=64)
parser.add_argument("--epochs_ssl",  type=int, default=10)
parser.add_argument("--epochs_sup",  type=int, default=40)
parser.add_argument("--device",      default="cuda")
parser.add_argument("--img_small",   type=int, default=224)
parser.add_argument("--img_big",     type=int, default=384)
parser.add_argument("--stride_small",type=int, default=160)
parser.add_argument("--stride_big",  type=int, default=256)
parser.add_argument("--folds",       type=int, default=5)
args = parser.parse_args()

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEV = torch.device(args.device if torch.cuda.is_available() else "cpu")
AMP = (DEV.type == "cuda")

os.makedirs("checkpoints", exist_ok=True)

# =============================================================================
# 2) Tile?cropping helper
# =============================================================================
def tile_crop(img: Image.Image, size: int, stride: int) -> List[Image.Image]:
    """
    Exhaustive tiling of a PIL.Image into patches of shape (size�size) 
    with given stride, including edge/corner patches.
    """
    w, h = img.size
    out = []
    # grid over top/left positions
    for top in range(0, h - size + 1, stride):
        for left in range(0, w - size + 1, stride):
            out.append(img.crop((left, top, left + size, top + size)))
    # bottom strip (if remainder)
    if (h - size) % stride:
        for left in range(0, w - size + 1, stride):
            out.append(img.crop((left, h - size, left + size, h)))
    # right strip
    if (w - size) % stride:
        for top in range(0, h - size + 1, stride):
            out.append(img.crop((w - size, top, w, top + size)))
    # bottom?right corner
    if (h - size) % stride and (w - size) % stride:
        out.append(img.crop((w - size, h - size, w, h)))
    return out

# =============================================================================
# 3) Data Discovery
# =============================================================================
def discover(root: str) -> Tuple[List[str], List[int], List[float]]:
    """
    Finds all frame_*.png under material_i/cycle_j/ 
    Returns (all_frame_paths, labels, dummy_depths).
    Label = (material_i_id - 1). Depths are dummy (set=1.0). 
    """
    imgs, lbls, depths = [], [], []
    for mat in sorted(glob.glob(os.path.join(root, "material_*"))):
        tail = mat.split("_")[-1]
        if not tail.isdigit():
            continue
        cls = int(tail) - 1
        for cycle_folder in glob.glob(os.path.join(mat, "cycle_*")):
            # expect 12 frames named frame_01.png?frame_12.png
            frames = sorted(glob.glob(os.path.join(cycle_folder, "frame_*.png")))
            for fr in frames:
                imgs.append(fr)
                lbls.append(cls)
                depths.append(1.0)  # dummy, could be actual depth if available
    return imgs, lbls, depths

IMGS, LABELS, DEPTHS = discover(args.root)
NCLS = len(set(LABELS))
print(f"Loaded {len(IMGS)} frames across {NCLS} classes")

# =============================================================================
# 4) Augmentations
# =============================================================================
MEAN, STD = IMAGENET_NORMALIZE["mean"], IMAGENET_NORMALIZE["std"]

def get_train_transforms_small(size: int):
    return T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.02,0.1), ratio=(0.3,3.3)),
        T.Normalize(MEAN, STD),
    ])

def get_train_transforms_big(size: int):
    return T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.02,0.1), ratio=(0.3,3.3)),
        T.Normalize(MEAN, STD),
    ])

def get_val_transforms(size: int):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

A_SM = get_train_transforms_small(args.img_small)
A_BG = get_train_transforms_big(args.img_big)
V_SM = get_val_transforms(args.img_small)
V_BG = get_val_transforms(args.img_big)

# =============================================================================
# 5) Dataset Classes
# =============================================================================
class SSLTiles(Dataset):
    """
    For SSL pretraining: returns two independently augmented small?crop views of the same image.
    We ignore the big scale in SSL for simplicity.
    """
    def __init__(self, paths: List[str], size: int, stride: int):
        self.paths = paths
        self.size = size
        self.stride = stride
        self.tf = get_train_transforms_small(size)

    def __len__(self):
        return len(self.paths)

    def _random_crop(self, img: Image.Image):
        w, h = img.size
        max_x = w - self.size
        max_y = h - self.size
        if max_x <= 0 or max_y <= 0:
            # fallback to center?crop
            return img.resize((self.size, self.size))
        xs = list(range(0, max_x + 1, self.stride))
        ys = list(range(0, max_y + 1, self.stride))
        x = random.choice(xs)
        y = random.choice(ys)
        return img.crop((x, y, x + self.size, y + self.size))

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        t1 = self._random_crop(img)
        t2 = self._random_crop(img)
        return self.tf(t1), self.tf(t2)

class SupTiles(Dataset):
    """
    For supervised training: returns:
      sm: (6, 3, H_sm, W_sm)  ? six random small tiles
      bg: (4, 3, H_bg, W_bg)  ? four random big tiles
      vol: (12, 3, H_sm, W_sm) ? the entire 12?frame volume (resized + augment per?frame)
      label: int
      depth_code: (2,)
      cycle_id: str (e.g. 'cycle_17') for voting
    """
    def __init__(self, paths: List[str], lbls: List[int], depths: List[float], train: bool = True):
        self.records = list(zip(paths, lbls, depths))
        self.train = train
        self.tf_sm = A_SM if train else V_SM
        self.tf_bg = A_BG if train else V_BG
        self.tf_frame = T.Compose([
            T.Resize(args.img_small),
            T.RandomRotation(2) if train else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(MEAN, STD)
        ])
        self.size_sm, self.stride_sm = args.img_small, args.stride_small
        self.size_bg, self.stride_bg = args.img_big,   args.stride_big

    def __len__(self):
        return len(self.records)

    def _random_tiles(self, img: Image.Image, size: int, stride: int, k: int) -> torch.Tensor:
        tiles = tile_crop(img, size, stride)
        if len(tiles) == 0:
            # fallback: resize + center?crop
            img_resized = img.resize((size, size))
            tile = img_resized
            if size == self.size_sm:
                return torch.stack([self.tf_sm(tile) for _ in range(k)])
            else:
                return torch.stack([self.tf_bg(tile) for _ in range(k)])
        sampled = random.sample(tiles, min(k, len(tiles)))
        if size == self.size_sm:
            return torch.stack([self.tf_sm(t) for t in sampled])
        else:
            return torch.stack([self.tf_bg(t) for t in sampled])

    def __getitem__(self, idx):
        path, label, depth = self.records[idx]
        # path looks like .../material_i/cycle_j/frame_01.png
        frame_folder = str(Path(path).parent)  # cycle_j folder
        # 1) load 12 frames as volume
        volume_frames = []
        for i in range(1, 13):
            fpath = os.path.join(frame_folder, f"frame_{i:02d}.png")
            img = Image.open(fpath).convert("RGB")
            img_t = self.tf_frame(img)  # (3, H_sm, W_sm)
            volume_frames.append(img_t.unsqueeze(0))  # (1, 3, H, W)
        vol = torch.cat(volume_frames, dim=0)  # (12, 3, H_sm, W_sm)

        # 2) random?tile small: 6 crops
        orig_img = Image.open(path).convert("RGB")
        sm = self._random_tiles(orig_img, self.size_sm, self.stride_sm, 6)  # (6,3,H_sm,W_sm)

        # 3) random?tile big: 4 crops
        bg = self._random_tiles(orig_img, self.size_bg, self.stride_bg, 4)  # (4,3,H_bg,W_bg)

        # 4) depth encoding (sin, cos)
        d_norm = depth / 50.0  # assume max depth ~50 mm
        depth_code = torch.tensor([math.sin(2 * math.pi * d_norm), 
                                   math.cos(2 * math.pi * d_norm)], 
                                  dtype=torch.float32)

        # 5) cycle?ID (for voting): 'cycle_j'
        cycle_id = Path(path).parent.name

        return sm, bg, vol, label, depth_code, cycle_id

# =============================================================================
# 6) Model Definition
# =============================================================================
class DualPyramidWith3DAndMetric(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # 2D backbone: EfficientNet?B3
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_dim = self.backbone.classifier[1].in_features  # 1536
        self.backbone.classifier = nn.Identity()

        # FiLM depth fusion
        self.depth_mlp = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim * 2),  # produce ? and ? each of length 1536
        )

        # Local attention on 6 small tiles
        self.local_mha = nn.MultiheadAttention(embed_dim=in_dim * 2, num_heads=8, batch_first=True)
        # Global attention on 4 big tiles
        self.global_mha = nn.MultiheadAttention(embed_dim=in_dim * 2, num_heads=8, batch_first=True)

        # 3D CNN for volume embedding
        # Input: (B, 12, 3, H_sm, W_sm) ? reorder to (B,3,12,H_sm,W_sm)
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # (32, 6, H/2, W/2)

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # (64, 3, H/4, W/4)

            nn.AdaptiveAvgPool3d((1, 1, 1)),  # (64, 1, 1, 1)
            nn.Flatten(),                     # (64,)
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Final fusion head: local (3072) + global (3072) + 3D (512) = 6656 total
        fusion_dim = (in_dim * 2) * 2 + 512
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def encode_2d_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of tiles through EfficientNet?B3, then simply replicate each feature 
        into a (mean, max) pair so we end up with (in_dim*2). 
        tiles: (B_tiles, 3, H, W)
        Returns: (B_tiles, in_dim*2)
        """
        f = self.backbone(tiles)  # (B_tiles, in_dim)
        return torch.cat([f, f], dim=1)  # (B_tiles, in_dim*2)

    def forward(self, sm: torch.Tensor, bg: torch.Tensor, vol: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sm:  (B, 6,    3, H_sm, W_sm)
        bg:  (B, 4,    3, H_bg, W_bg)
        vol: (B, 12,   3, H_sm, W_sm)
        depth: (B, 2)
        Returns:
          logits:    (B, num_classes)
          embedding: (B, fusion_dim)  # before final classification (for metric loss)
        """
        B = sm.size(0)

        # === 1) 2D Small tiles ===
        sm_flat = sm.view(-1, 3, sm.size(3), sm.size(4))  # (B*6, 3, H_sm, W_sm)
        enc_sm = self.encode_2d_tiles(sm_flat)            # (B*6, in_dim*2)

        # === 2) 2D Big tiles ===
        bg_flat = bg.view(-1, 3, bg.size(3), bg.size(4))  # (B*4, 3, H_bg, W_bg)
        enc_bg = self.encode_2d_tiles(bg_flat)            # (B*4, in_dim*2)

        # === 3) 3D volume ===
        # reorder for Conv3d: want (B, C=3, D=12, H_sm, W_sm)
        vol_3d = vol.permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, 12, H_sm, W_sm)
        emb_3d = self.conv3d(vol_3d)                      # (B, 512)

        # === 4) Depth FiLM fusion ===
        film_params = self.depth_mlp(depth)  # (B, in_dim*2)
        gamma, beta = film_params.chunk(2, dim=1)  # (B, in_dim) each

        # Apply to small?tile embeddings:
        half = enc_sm.size(1) // 2
        e1_sm, e2_sm = enc_sm[:, :half], enc_sm[:, half:]
        gamma_sm = gamma.repeat_interleave(6, dim=0)  # (B*6, in_dim)
        beta_sm  = beta.repeat_interleave(6, dim=0)
        e1_sm = gamma_sm * e1_sm + beta_sm
        e2_sm = gamma_sm * e2_sm + beta_sm
        enc_sm = torch.cat([e1_sm, e2_sm], dim=1)  # (B*6, in_dim*2)

        # Apply to big?tile embeddings:
        half = enc_bg.size(1) // 2
        e1_bg, e2_bg = enc_bg[:, :half], enc_bg[:, half:]
        gamma_bg = gamma.repeat_interleave(4, dim=0)  # (B*4, in_dim)
        beta_bg  = beta.repeat_interleave(4, dim=0)
        e1_bg = gamma_bg * e1_bg + beta_bg
        e2_bg = gamma_bg * e2_bg + beta_bg
        enc_bg = torch.cat([e1_bg, e2_bg], dim=1)  # (B*4, in_dim*2)

        # === 5) Reshape into token sequences ===
        local_tokens = enc_sm.view(B, 6, -1)   # (B, 6, in_dim*2)
        global_tokens = enc_bg.view(B, 4, -1)  # (B, 4, in_dim*2)

        # === 6) Multihead attention over local & global tokens ===
        local_attn, _  = self.local_mha(local_tokens,  local_tokens,  local_tokens)
        global_attn, _ = self.global_mha(global_tokens, global_tokens, global_tokens)

        local_pooled  = local_attn.mean(dim=1)   # (B, in_dim*2)
        global_pooled = global_attn.mean(dim=1)  # (B, in_dim*2)

        # === 7) Concatenate 2D + 3D ===
        fusion = torch.cat([local_pooled, global_pooled, emb_3d], dim=1)  # (B, fusion_dim)

        # === 8) Final classification head ===
        logits = self.head(fusion)  # (B, num_classes)

        return logits, fusion  # return fusion as the ?embedding? for triplet loss

# =============================================================================
# 7) SSL Pretraining (2D only, on small?crop contrastive)
# =============================================================================
def simclr_pretrain(model: DualPyramidWith3DAndMetric, paths: List[str]):
    """
    Pretrain ?model.backbone? via SimCLR on small?crop augmentations.
    We ignore the 3D path during SSL.
    """
    class SSLSingleScale(Dataset):
        def __init__(self, paths, size, stride):
            self.paths = paths
            self.size = size
            self.stride = stride
            self.tf = get_train_transforms_small(size)
        def __len__(self):
            return len(self.paths)
        def _random_crop(self, img: Image.Image):
            w, h = img.size
            max_x, max_y = w - self.size, h - self.size
            if max_x <= 0 or max_y <= 0:
                return img.resize((self.size, self.size))
            xs = list(range(0, max_x + 1, self.stride))
            ys = list(range(0, max_y + 1, self.stride))
            x = random.choice(xs)
            y = random.choice(ys)
            return img.crop((x, y, x + self.size, y + self.size))
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            t1 = self._random_crop(img)
            t2 = self._random_crop(img)
            return self.tf(t1), self.tf(t2)

    ssl_ds = SSLSingleScale(paths, args.img_small, args.stride_small)
    dl = DataLoader(ssl_ds, batch_size=args.batch_ssl, shuffle=True,
                    num_workers=8, drop_last=True, pin_memory=True)

    # Freeze everything except model.backbone
    deactivate_requires_grad(model.local_mha)
    deactivate_requires_grad(model.global_mha)
    deactivate_requires_grad(model.depth_mlp)
    deactivate_requires_grad(model.conv3d)
    deactivate_requires_grad(model.head)
    activate_requires_grad(model.backbone)

    # We need to know the backbone?s output dimension. Because `model.backbone.classifier` was replaced
    # with Identity, we re?instantiate a fresh EfficientNet_B3 to read off classifier[1].in_features:
    tmp_e3 = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    backbone_dim = tmp_e3.classifier[1].in_features  # 1536
    del tmp_e3

    # Projector head for contrastive
    proj = nn.Sequential(
        nn.Linear(backbone_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 128),
    ).to(DEV)

    base_opt = torch.optim.SGD(proj.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-6)
    opt = LARS(base_opt) if LARS else base_opt
    loss_fn = NTXentLoss().to(DEV)

    # Contrastive training
    for ep in range(1, args.epochs_ssl + 1):
        pbar = tqdm(dl, desc=f"SSL Epoch {ep}/{args.epochs_ssl}")
        for v1, v2 in pbar:
            v1 = v1.to(DEV)
            v2 = v2.to(DEV)
            with torch.no_grad(), torch.autocast(device_type=DEV.type, enabled=AMP):
                h1 = model.backbone(v1)  # (B, 1536)
                h2 = model.backbone(v2)
            z1 = proj(h1.float())
            z2 = proj(h2.float())
            loss = loss_fn(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Save backbone weights
    torch.save(model.state_dict(), "checkpoints/ssl_b3_rgb_with3d.pt")
    # Unfreeze all
    activate_requires_grad(model.local_mha)
    activate_requires_grad(model.global_mha)
    activate_requires_grad(model.depth_mlp)
    activate_requires_grad(model.conv3d)
    activate_requires_grad(model.head)

# =============================================================================
# 8) Supervised Training with Auxiliary Triplet Loss
# =============================================================================
def sup_train(model: DualPyramidWith3DAndMetric,
              train_ds: SupTiles,
              val_ds: SupTiles) -> Dict:
    """
    Returns the best state_dict. Uses cycle?level 
    voting plus a triplet margin loss on the 1024?D embedding.
    """
    # 8.1) Balanced sampler across classes
    label_counts = {}
    for _, _, _, lbl, _, _ in train_ds:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    weights = [1.0 / label_counts[lbl] for *_, lbl, _, _ in train_ds]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dl_tr = DataLoader(train_ds, batch_size=args.batch_sup, sampler=sampler,
                       num_workers=6, pin_memory=True)
    dl_va = DataLoader(val_ds, batch_size=args.batch_sup, shuffle=False,
                       num_workers=4, pin_memory=True)

    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
                                          steps_per_epoch=len(dl_tr),
                                          epochs=args.epochs_sup)
    ce_loss = lambda x, y: nn.functional.cross_entropy(x, y, label_smoothing=0.05)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)  # Euclidean margin
    f1_metric = MulticlassF1Score(num_classes=NCLS, average="macro").to(DEV)

    best_score, best_sd = 0.0, None

    for ep in range(1, args.epochs_sup + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Sup Ep {ep}/{args.epochs_sup}")
        for sm, bg, vol, label, depth_code, _ in pbar:
            sm = sm.to(DEV)         # (B,6,3,H_sm,W_sm)
            bg = bg.to(DEV)         # (B,4,3,H_bg,W_bg)
            vol = vol.to(DEV)       # (B,12,3,H_sm,W_sm)
            label = label.to(DEV)   # (B,)
            depth_code = depth_code.to(DEV)  # (B,2)

            logits, embedding = model(sm, bg, vol, depth_code)  # (B,NCLS), (B, fusion_dim)
            loss_ce = ce_loss(logits, label)

            # Build one random triplet per anchor
            B = label.size(0)
            if B >= 3:
                loss_tri = torch.tensor(0.0, device=DEV)
                valid_tri = 0
                for i in range(B):
                    anchor = embedding[i : i+1]
                    pos_indices = (label == label[i]).nonzero(as_tuple=False).squeeze().tolist()
                    neg_indices = (label != label[i]).nonzero(as_tuple=False).squeeze().tolist()
                    if isinstance(pos_indices, list):
                        pos_choices = [p for p in pos_indices if p != i]
                    else:
                        pos_choices = []
                    if len(pos_choices) == 0 or len(neg_indices) == 0:
                        continue
                    pos = random.choice(pos_choices)
                    neg = random.choice(neg_indices)
                    pos_emb = embedding[pos : pos+1]
                    neg_emb = embedding[neg : neg+1]
                    loss_tri = loss_tri + triplet_loss(anchor, pos_emb, neg_emb)
                    valid_tri += 1
                if valid_tri > 0:
                    loss_tri = loss_tri / valid_tri
                else:
                    loss_tri = torch.tensor(0.0, device=DEV)
            else:
                loss_tri = torch.tensor(0.0, device=DEV)

            loss = loss_ce + 0.2 * loss_tri  # weight ?=0.2 for metric loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        # Validation with cycle?level voting
        model.eval()
        cycle_preds: Dict[str, List[torch.Tensor]] = {}
        cycle_labels: Dict[str, int] = {}

        with torch.no_grad(), torch.autocast(device_type=DEV.type, enabled=AMP):
            for sm, bg, vol, label, depth_code, cycle_id in dl_va:
                sm = sm.to(DEV)
                bg = bg.to(DEV)
                vol = vol.to(DEV)
                depth_code = depth_code.to(DEV)
                logit, _ = model(sm, bg, vol, depth_code)
                probs = logit.softmax(dim=1).cpu()  # (B, NCLS)
                for i, cid in enumerate(cycle_id):
                    cstr = cid  # already string
                    if cstr not in cycle_preds:
                        cycle_preds[cstr] = []
                        cycle_labels[cstr] = label[i].item()
                    cycle_preds[cstr].append(probs[i].unsqueeze(0))

        # Compute cycle?level F1
        all_preds, all_truths = [], []
        for cid, pred_list in cycle_preds.items():
            avg = torch.cat(pred_list, dim=0).mean(dim=0, keepdim=True)  # (1, NCLS)
            pred_label = avg.argmax(dim=1)
            true_label = torch.tensor([cycle_labels[cid]], device=DEV)
            all_preds.append(pred_label)
            all_truths.append(true_label)
        all_preds = torch.cat(all_preds, dim=0)
        all_truths = torch.cat(all_truths, dim=0)
        score = f1_metric(all_preds, all_truths).item()
        print(f"  [Val Cycle?level] F1={score:.3f}")

        if score > best_score:
            best_score = score
            best_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_sd

# =============================================================================
# 9) K?fold Orchestration & Final Report
# =============================================================================
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
ALL_CYCLE_PRED, ALL_CYCLE_TRUE = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(IMGS, LABELS), start=1):
    print(f"\n== Fold {fold}/{args.folds} ==")
    train_paths = [IMGS[i] for i in train_idx] 
    train_labels = [LABELS[i] for i in train_idx]
    train_depths = [DEPTHS[i] for i in train_idx]
    val_paths   = [IMGS[i] for i in val_idx]
    val_labels  = [LABELS[i] for i in val_idx]
    val_depths  = [DEPTHS[i] for i in val_idx]

    model = DualPyramidWith3DAndMetric(NCLS).to(DEV)

    # 9.1) SSL Stage
    if args.stage in ("ssl", "all"):
        print("[*] Starting SSL pretraining ?")
        simclr_pretrain(model, train_paths)
        print("[*] SSL pretraining complete")

    # 9.2) Supervised Stage
    if args.stage in ("finetune", "all"):
        print("[*] Starting supervised finetuning ?")
        train_ds = SupTiles(train_paths, train_labels, train_depths, train=True)
        val_ds   = SupTiles(val_paths, val_labels, val_depths, train=False)
        best_state = sup_train(model, train_ds, val_ds)
        model.load_state_dict(best_state)
        print("[*] Supervised finetuning complete")

    # 9.3) Save fold model
    torch.save(model.state_dict(), f"checkpoints/fold{fold}_b3_3d.pt")

    # 9.4) Final evaluation on validation fold (cycle?level)
    cycle_preds: Dict[str, List[torch.Tensor]] = {}
    cycle_labels: Dict[str, int] = {}
    val_ds_infer = SupTiles(val_paths, val_labels, val_depths, train=False)
    dl_va_final = DataLoader(val_ds_infer, batch_size=args.batch_sup, 
                             shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=DEV.type, enabled=AMP):
        for sm, bg, vol, label, depth_code, cycle_id in dl_va_final:
            sm = sm.to(DEV)
            bg = bg.to(DEV)
            vol = vol.to(DEV)
            depth_code = depth_code.to(DEV)
            logit, _ = model(sm, bg, vol, depth_code)
            probs = logit.softmax(dim=1).cpu()
            for i, cid in enumerate(cycle_id):
                cstr = cid
                if cstr not in cycle_preds:
                    cycle_preds[cstr] = []
                    cycle_labels[cstr] = label[i].item()
                cycle_preds[cstr].append(probs[i].unsqueeze(0))

    # gather final cycle?level preds
    for cid, plist in cycle_preds.items():
        avg = torch.cat(plist, dim=0).mean(dim=0, keepdim=True)  # (1, NCLS)
        pl = avg.argmax(dim=1).item()
        tl = cycle_labels[cid]
        ALL_CYCLE_PRED.append(pl)
        ALL_CYCLE_TRUE.append(tl)

# 9.5) Global report
print("\n===== Final 5?Fold Cycle?Level Results =====")
PRED = np.array(ALL_CYCLE_PRED)
TRUE = np.array(ALL_CYCLE_TRUE)
print(classification_report(TRUE, PRED, digits=4))
print(confusion_matrix(TRUE, PRED))
