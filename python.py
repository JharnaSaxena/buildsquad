# ============================================================
# BUILDSQUAD - DUALITY AI HACKATHON
# SegFormer B2 + B0 Ensemble Semantic Segmentation
# Team: Jharna Saxena, Palak Saxena
# ============================================================

import os, shutil, gc, torch, time, random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm
from transformers import SegformerForSemanticSegmentation
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ── STEP 1: CHECK GPU ──────────────────────────────────────
print("="*55)
print("STEP 1: GPU CHECK")
print("="*55)
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
else:
    print('❌ No GPU! Runtime → Change runtime type → T4 GPU')
    raise SystemExit("Need GPU!")

# ── STEP 2: INSTALL LIBRARIES ──────────────────────────────
print("\n" + "="*55)
print("STEP 2: INSTALLING LIBRARIES")
print("="*55)
os.system("pip install -q transformers==4.41.0 albumentations==1.4.3 timm accelerate")
print('✅ Done!')

# ── STEP 3: MOUNT DRIVE ────────────────────────────────────
print("\n" + "="*55)
print("STEP 3: MOUNT DRIVE")
print("="*55)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

GULU = '/content/drive/MyDrive/gulu'
DRIVE_OUT = '/content/drive/MyDrive/duality_results'
os.makedirs('/content/runs', exist_ok=True)
os.makedirs('/content/runs/predictions', exist_ok=True)
os.makedirs(DRIVE_OUT, exist_ok=True)

for sub in ['train/Color_Images','train/Segmentation',
            'val/Color_Images','val/Segmentation']:
    path = f'{GULU}/{sub}'
    count = len(os.listdir(path)) if os.path.exists(path) else 0
    print(f'  {"✅" if count>0 else "❌"} {sub}: {count} files')

# ── STEP 4: FIX FILENAMES ──────────────────────────────────
print("\n" + "="*55)
print("STEP 4: FIX FILENAMES")
print("="*55)
def remove_spaces(folder):
    if not os.path.exists(folder): return
    renamed = 0
    for fname in os.listdir(folder):
        if ' ' in fname:
            os.rename(os.path.join(folder, fname),
                      os.path.join(folder, fname.replace(' ', '_')))
            renamed += 1
    print(f'  ✅ {os.path.basename(folder)}: {renamed} renamed')

remove_spaces(f'{GULU}/train/Color_Images')
remove_spaces(f'{GULU}/train/Segmentation')
remove_spaces(f'{GULU}/val/Color_Images')
remove_spaces(f'{GULU}/val/Segmentation')

# ── STEP 5: CONFIG ─────────────────────────────────────────
print("\n" + "="*55)
print("STEP 5: CONFIG")
print("="*55)
CONFIG = {
    'train_img_dir':  f'{GULU}/train/Color_Images',
    'train_mask_dir': f'{GULU}/train/Segmentation',
    'val_img_dir':    f'{GULU}/val/Color_Images',
    'val_mask_dir':   f'{GULU}/val/Segmentation',
    'output_dir':     '/content/runs',
    'drive_dir':      DRIVE_OUT,
    'model_1':        'nvidia/mit-b2',
    'model_2':        'nvidia/mit-b0',
    'num_classes':    10,
    'img_size':       320,
    'batch_size':     2,
    'num_epochs':     50,
    'lr':             6e-5,
    'weight_decay':   0.01,
    'patience':       10,
    'num_workers':    0,
    'device':         'cuda',
}
print(f'✅ Config ready!')
print(f'   model_1 : {CONFIG["model_1"]}')
print(f'   model_2 : {CONFIG["model_2"]}')
print(f'   img_size: {CONFIG["img_size"]}')

# ── STEP 6: CLASSES ────────────────────────────────────────
CLASS_MAP = {
    100:0, 200:1, 300:2, 500:3, 550:4,
    600:5, 700:6, 800:7, 7100:8, 10000:9
}
IDX_TO_NAME = {
    0:'Trees', 1:'Lush Bushes', 2:'Dry Grass', 3:'Dry Bushes',
    4:'Ground Clutter', 5:'Flowers', 6:'Logs', 7:'Rocks',
    8:'Landscape', 9:'Sky'
}
CLASS_COLORS = {
    0:(34,139,34), 1:(0,200,100), 2:(210,180,140), 3:(139,90,43),
    4:(128,128,128), 5:(255,20,147), 6:(101,67,33),
    7:(169,169,169), 8:(205,133,63), 9:(135,206,235),
}

def remap_mask(mask_np):
    out = np.full_like(mask_np, 255, dtype=np.uint8)
    for orig, new_id in CLASS_MAP.items():
        out[mask_np == orig] = new_id
    return out

print('✅ Classes ready!')

# ── STEP 7: DATASET ────────────────────────────────────────
print("\n" + "="*55)
print("STEP 7: DATASET")
print("="*55)

def get_transforms(split='train', sz=320):
    if split == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(sz,sz), scale=(0.6,1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.HueSaturationValue(15, 25, 20, p=0.4),
            A.GaussianBlur(blur_limit=(3,7), p=0.2),
            A.GaussNoise(p=0.2),
            A.RandomShadow(p=0.3),
            A.CoarseDropout(p=0.2),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

def cutmix(img1, mask1, img2, mask2):
    _, H, W = img1.shape
    lam   = random.uniform(0.2, 0.8)
    cut_w = int(W * (1-lam)**0.5)
    cut_h = int(H * (1-lam)**0.5)
    cx = random.randint(0, max(1, W-cut_w))
    cy = random.randint(0, max(1, H-cut_h))
    img1  = img1.clone(); mask1 = mask1.clone()
    img1[:, cy:cy+cut_h, cx:cx+cut_w] = img2[:, cy:cy+cut_h, cx:cx+cut_w]
    mask1[cy:cy+cut_h, cx:cx+cut_w]   = mask2[cy:cy+cut_h, cx:cx+cut_w]
    return img1, mask1

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, split='train', sz=320, use_cutmix=False):
        self.img_dir    = img_dir
        self.mask_dir   = mask_dir
        self.tf         = get_transforms(split, sz)
        self.use_cutmix = use_cutmix and split == 'train'
        all_imgs = sorted([f for f in os.listdir(img_dir)
                           if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.imgs=[]; self.masks=[]; skipped=0
        for fname in all_imgs:
            base = os.path.splitext(fname)[0]
            mp = next((os.path.join(mask_dir, base+e)
                       for e in ['.png','.jpg','.jpeg']
                       if os.path.exists(os.path.join(mask_dir, base+e))), None)
            if mp: self.imgs.append(fname); self.masks.append(mp)
            else: skipped+=1
        print(f'  {split}: {len(self.imgs)} pairs, {skipped} skipped')
        assert len(self.imgs) > 0

    def __len__(self): return len(self.imgs)

    def load_one(self, idx):
        img  = np.array(Image.open(os.path.join(self.img_dir, self.imgs[idx])).convert('RGB'))
        mask = np.array(Image.open(self.masks[idx]))
        if mask.ndim == 3: mask = mask[:,:,0]
        mask = remap_mask(mask)
        aug  = self.tf(image=img, mask=mask)
        img_t = aug['image']
        msk_t = aug['mask']
        if isinstance(msk_t, torch.Tensor):
            msk_t = msk_t.detach().clone().long()
        else:
            msk_t = torch.tensor(msk_t, dtype=torch.long)
        return img_t, msk_t

    def __getitem__(self, i):
        img, mask = self.load_one(i)
        if self.use_cutmix and random.random() < 0.4:
            j = random.randint(0, len(self.imgs)-1)
            img2, mask2 = self.load_one(j)
            img, mask   = cutmix(img, mask, img2, mask2)
        return img, mask

train_ds = SegDataset(CONFIG['train_img_dir'], CONFIG['train_mask_dir'],
                      'train', CONFIG['img_size'], use_cutmix=True)
val_ds   = SegDataset(CONFIG['val_img_dir'],   CONFIG['val_mask_dir'],
                      'val',   CONFIG['img_size'], use_cutmix=False)
train_loader = DataLoader(train_ds, CONFIG['batch_size'], shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   CONFIG['batch_size'], shuffle=False,
                          num_workers=0, pin_memory=True)
print('✅ DataLoaders ready!')

# ── STEP 8: LOSS ───────────────────────────────────────────
device = torch.device('cuda')

class CombinedLoss(nn.Module):
    def __init__(self, weights=None, nc=10, alpha=0.6):
        super().__init__()
        self.nc=nc; self.alpha=alpha
        self.ce=nn.CrossEntropyLoss(weight=weights, ignore_index=255)

    def dice(self, logits, targets):
        probs=torch.softmax(logits,1); total=0; n=0
        for c in range(self.nc):
            pc=probs[:,c]; tc=(targets==c).float()
            if tc.sum()==0: continue
            inter=(pc*tc).sum()
            total+=1-(2*inter+1e-6)/(pc.sum()+tc.sum()+1e-6); n+=1
        return total/max(n,1)

    def forward(self, logits, targets):
        return self.alpha*self.ce(logits,targets)+(1-self.alpha)*self.dice(logits,targets)

weights    = torch.ones(10)
weights[5] = 3.0
weights[6] = 3.0
weights[4] = 1.5
criterion  = CombinedLoss(weights.to(device))
print('✅ Loss ready!')

# ── STEP 9: TRAINING FUNCTION ──────────────────────────────
def train_model(model, optimizer, scaler, scheduler, name, save_path, drive_path):
    best_iou=0; patience_ct=0
    history={'tl':[],'vl':[],'ti':[],'vi':[]}
    print(f'\n🚀 Training {name}...')
    print('='*55)
    for epoch in range(1, CONFIG['num_epochs']+1):
        t0=time.time()
        model.train(); tloss=0
        train_inter=np.zeros(10); train_union=np.zeros(10)
        for imgs,masks in tqdm(train_loader, desc=f'Ep{epoch} Train', leave=False):
            imgs,masks=imgs.to(device),masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out=model(pixel_values=imgs).logits
                out=nn.functional.interpolate(out,size=masks.shape[-2:],
                                              mode='bilinear',align_corners=False)
                loss=criterion(out,masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            tloss+=loss.item()
            pred=out.argmax(1).detach().cpu().numpy().flatten()
            gt=masks.detach().cpu().numpy().flatten()
            valid=gt!=255; pred=pred[valid]; gt=gt[valid]
            for c in range(10):
                train_inter[c]+=((pred==c)&(gt==c)).sum()
                train_union[c]+=((pred==c)|(gt==c)).sum()
            del out,loss,pred,gt; torch.cuda.empty_cache()

        model.eval(); vloss=0
        val_inter=np.zeros(10); val_union=np.zeros(10)
        with torch.no_grad():
            for imgs,masks in tqdm(val_loader, desc=f'Ep{epoch} Val', leave=False):
                imgs,masks=imgs.to(device),masks.to(device)
                with torch.amp.autocast('cuda'):
                    out=model(pixel_values=imgs).logits
                    out=nn.functional.interpolate(out,size=masks.shape[-2:],
                                                  mode='bilinear',align_corners=False)
                    vloss+=criterion(out,masks).item()
                pred=out.argmax(1).detach().cpu().numpy().flatten()
                gt=masks.detach().cpu().numpy().flatten()
                valid=gt!=255; pred=pred[valid]; gt=gt[valid]
                for c in range(10):
                    val_inter[c]+=((pred==c)&(gt==c)).sum()
                    val_union[c]+=((pred==c)|(gt==c)).sum()
                del out,pred,gt; torch.cuda.empty_cache()

        scheduler.step()
        tl_=tloss/len(train_loader); vl_=vloss/len(val_loader)
        ti_ious=[train_inter[c]/train_union[c] for c in range(10) if train_union[c]>0]
        vi_ious=[val_inter[c]/val_union[c] for c in range(10) if val_union[c]>0]
        ti=float(np.mean(ti_ious)) if ti_ious else 0.0
        vi=float(np.mean(vi_ious)) if vi_ious else 0.0
        history['tl'].append(tl_); history['vl'].append(vl_)
        history['ti'].append(ti);  history['vi'].append(vi)
        gc.collect()
        print(f'Ep[{epoch:03d}] Loss:{vl_:.4f} mIoU:{vi:.4f} ({time.time()-t0:.0f}s)')
        if vi>best_iou:
            best_iou=vi; patience_ct=0
            torch.save({'epoch':epoch,'model':model.state_dict(),'val_iou':vi}, save_path)
            shutil.copy(save_path, drive_path)
            print(f'  ✅ Best {best_iou:.4f} saved to Drive!')
        else:
            patience_ct+=1
            if patience_ct>=CONFIG['patience']:
                print('  ⏹️ Early stop!'); break
    print(f'\n🎉 {name} done! Best mIoU: {best_iou:.4f}')
    return best_iou, history

# ── STEP 10: TRAIN B2 ──────────────────────────────────────
print("\n" + "="*55)
print("STEP 10: TRAINING B2")
print("="*55)
gc.collect(); torch.cuda.empty_cache()
print(f'GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')

model1 = SegformerForSemanticSegmentation.from_pretrained(
    CONFIG['model_1'], num_labels=10, ignore_mismatched_sizes=True,
    id2label={str(i):n for i,n in IDX_TO_NAME.items()},
    label2id={n:i for i,n in IDX_TO_NAME.items()},
).to(device)

optimizer1 = torch.optim.AdamW(model1.parameters(),
                                lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer1, T_max=CONFIG['num_epochs'], eta_min=1e-6)
scaler1 = torch.amp.GradScaler('cuda')

best_iou1, h1 = train_model(
    model1, optimizer1, scaler1, scheduler1,
    'SegFormer-B2',
    f"{CONFIG['output_dir']}/best_model1_b2.pt",
    f"{CONFIG['drive_dir']}/best_model1_b2.pt"
)
print(f'✅ B2 done! mIoU: {best_iou1:.4f}')

# ── STEP 11: TRAIN B0 ──────────────────────────────────────
print("\n" + "="*55)
print("STEP 11: TRAINING B0")
print("="*55)
del model1, optimizer1, scaler1, scheduler1
gc.collect(); torch.cuda.empty_cache()
print(f'GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')

model2 = SegformerForSemanticSegmentation.from_pretrained(
    CONFIG['model_2'], num_labels=10, ignore_mismatched_sizes=True,
    id2label={str(i):n for i,n in IDX_TO_NAME.items()},
    label2id={n:i for i,n in IDX_TO_NAME.items()},
).to(device)

optimizer2 = torch.optim.AdamW(model2.parameters(),
                                lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer2, T_max=CONFIG['num_epochs'], eta_min=1e-6)
scaler2 = torch.amp.GradScaler('cuda')

best_iou2, h2 = train_model(
    model2, optimizer2, scaler2, scheduler2,
    'SegFormer-B0',
    f"{CONFIG['output_dir']}/best_model2_b0.pt",
    f"{CONFIG['drive_dir']}/best_model2_b0.pt"
)
print(f'✅ B0 done! mIoU: {best_iou2:.4f}')
print(f'🏆 B2:{best_iou1:.4f} | B0:{best_iou2:.4f}')

# ── STEP 12: LOAD BOTH MODELS ──────────────────────────────
print("\n" + "="*55)
print("STEP 12: LOADING ENSEMBLE")
print("="*55)
del model2, optimizer2, scaler2, scheduler2
gc.collect(); torch.cuda.empty_cache()

model1 = SegformerForSemanticSegmentation.from_pretrained(
    CONFIG['model_1'], num_labels=10, ignore_mismatched_sizes=True,
    id2label={str(i):n for i,n in IDX_TO_NAME.items()},
    label2id={n:i for i,n in IDX_TO_NAME.items()},
).to(device)
model1.load_state_dict(torch.load(
    f"{CONFIG['output_dir']}/best_model1_b2.pt", map_location=device)['model'])
model1.eval()

model2 = SegformerForSemanticSegmentation.from_pretrained(
    CONFIG['model_2'], num_labels=10, ignore_mismatched_sizes=True,
    id2label={str(i):n for i,n in IDX_TO_NAME.items()},
    label2id={n:i for i,n in IDX_TO_NAME.items()},
).to(device)
model2.load_state_dict(torch.load(
    f"{CONFIG['output_dir']}/best_model2_b0.pt", map_location=device)['model'])
model2.eval()
print(f'✅ Both loaded!')

# ── STEP 13: ENSEMBLE PREDICTIONS ─────────────────────────
print("\n" + "="*55)
print("STEP 13: ENSEMBLE PREDICTIONS")
print("="*55)
tf_base = A.Compose([
    A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2()])
tf_flip = A.Compose([
    A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    A.HorizontalFlip(p=1.0),
    A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)), ToTensorV2()])

@torch.no_grad()
def get_probs(model, img_np):
    t   = tf_base(image=img_np)['image'].unsqueeze(0).to(device)
    out = nn.functional.interpolate(model(pixel_values=t).logits,
          (CONFIG['img_size'],CONFIG['img_size']),mode='bilinear',align_corners=False)
    p   = torch.softmax(out,1)
    t2  = tf_flip(image=img_np)['image'].unsqueeze(0).to(device)
    out2= nn.functional.interpolate(model(pixel_values=t2).logits,
          (CONFIG['img_size'],CONFIG['img_size']),mode='bilinear',align_corners=False)
    p   = (p + torch.softmax(torch.flip(out2,[-1]),1)) / 2
    return p

@torch.no_grad()
def ensemble_predict(img_np):
    return ((get_probs(model1,img_np)+get_probs(model2,img_np))/2)\
            .argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

val_imgs = sorted([f for f in os.listdir(CONFIG['val_img_dir'])
                   if f.lower().endswith(('.png','.jpg','.jpeg'))])
print(f'Running on {len(val_imgs)} images...')
for fname in tqdm(val_imgs):
    img_np = np.array(Image.open(f"{CONFIG['val_img_dir']}/{fname}").convert('RGB'))
    pred   = ensemble_predict(img_np)
    base   = os.path.splitext(fname)[0]
    Image.fromarray(pred).save(f"{CONFIG['output_dir']}/predictions/{base}_pred.png")
print('✅ Done!')

# ── STEP 14: METRICS & VISUALIZATION ──────────────────────
print("\n" + "="*55)
print("STEP 14: METRICS")
print("="*55)
val_imgs = sorted([f for f in os.listdir(CONFIG['val_img_dir'])
                   if f.lower().endswith(('.png','.jpg','.jpeg'))])

all_preds=[]; all_labels=[]
for fname in tqdm(val_imgs, desc='Computing mIoU'):
    base = os.path.splitext(fname)[0]
    mask_path = next((f"{CONFIG['val_mask_dir']}/{base}{e}"
                     for e in ['.png','.jpg','.jpeg']
                     if os.path.exists(f"{CONFIG['val_mask_dir']}/{base}{e}")), None)
    if not mask_path: continue
    img_np  = np.array(Image.open(f"{CONFIG['val_img_dir']}/{fname}").convert('RGB'))
    mask_np = np.array(Image.open(mask_path))
    if mask_np.ndim==3: mask_np=mask_np[:,:,0]
    mask_np = remap_mask(mask_np)
    pred = ensemble_predict(img_np)
    pred_img = Image.fromarray(pred).resize(
        (mask_np.shape[1], mask_np.shape[0]), Image.NEAREST)
    all_preds.append(np.array(pred_img).flatten())
    all_labels.append(mask_np.flatten())

all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
valid      = all_labels != 255
all_preds  = all_preds[valid]
all_labels = all_labels[valid]

# Per class IoU
ious = {}
for c in range(10):
    p=all_preds==c; t=all_labels==c
    inter=(p&t).sum(); union=(p|t).sum()
    ious[IDX_TO_NAME[c]] = inter/union if union>0 else 0

vals = list(ious.values())
ensemble_miou = float(np.mean(vals))

# Confusion matrix
names = [IDX_TO_NAME[i] for i in range(10)]
cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
cm_norm = cm.astype(float)/cm.sum(axis=1,keepdims=True).clip(min=1)

fig,ax=plt.subplots(figsize=(13,10))
sns.heatmap(cm_norm,annot=True,fmt='.2f',cmap='Blues',
            xticklabels=names,yticklabels=names,ax=ax)
ax.set_title('Confusion Matrix — Ensemble',fontsize=14)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.xticks(rotation=35,ha='right')
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/confusion_matrix.png",dpi=150)
shutil.copy(f"{CONFIG['output_dir']}/confusion_matrix.png",
            f"{CONFIG['drive_dir']}/confusion_matrix.png")
plt.show()

# Per class IoU bar chart
colors_ = [[c/255 for c in CLASS_COLORS[i]] for i in range(10)]
fig,ax  = plt.subplots(figsize=(13,5))
bars    = ax.bar(list(ious.keys()),vals,color=colors_,edgecolor='black')
ax.set_ylim(0,1.1)
ax.set_title(f'Per-Class IoU | Ensemble mIoU:{ensemble_miou:.4f}',fontsize=14)
ax.set_xticklabels(list(ious.keys()),rotation=30,ha='right')
for bar,val in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
            f'{val:.3f}',ha='center',va='bottom',fontsize=10,fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/per_class_iou.png",dpi=150)
shutil.copy(f"{CONFIG['output_dir']}/per_class_iou.png",
            f"{CONFIG['drive_dir']}/per_class_iou.png")
plt.show()

# Segmentation visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sample_imgs = val_imgs[:2]
for row, fname in enumerate(sample_imgs):
    img_np  = np.array(Image.open(f"{CONFIG['val_img_dir']}/{fname}").convert('RGB'))
    base    = os.path.splitext(fname)[0]
    mask_path = next((f"{CONFIG['val_mask_dir']}/{base}{e}"
                     for e in ['.png','.jpg','.jpeg']
                     if os.path.exists(f"{CONFIG['val_mask_dir']}/{base}{e}")), None)
    mask_np = remap_mask(np.array(Image.open(mask_path))[:,:,0] if np.array(Image.open(mask_path)).ndim==3 else np.array(Image.open(mask_path)))
    pred    = ensemble_predict(img_np)
    pred_color = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for idx, color in CLASS_COLORS.items():
        pred_color[pred==idx] = color
    overlay = (img_np * 0.5 + pred_color * 0.5).astype(np.uint8)
    axes[row,0].imshow(img_np);       axes[row,0].set_title('Image');   axes[row,0].axis('off')
    axes[row,1].imshow(pred_color);   axes[row,1].set_title('Mask');    axes[row,1].axis('off')
    axes[row,2].imshow(overlay);      axes[row,2].set_title('Overlay'); axes[row,2].axis('off')
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/segmentation_viz.png",dpi=150)
shutil.copy(f"{CONFIG['output_dir']}/segmentation_viz.png",
            f"{CONFIG['drive_dir']}/segmentation_viz.png")
plt.show()
import matplotlib.pyplot as plt, random

train_imgs = sorted(os.listdir(CONFIG['train_img_dir']))
print(f'Train: {len(train_imgs)} images')

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for row, fname in enumerate(random.sample(train_imgs, 2)):
    base = os.path.splitext(fname)[0]
    img  = np.array(Image.open(f"{CONFIG['train_img_dir']}/{fname}").convert('RGB'))
    for ext in ['.png','.jpg','.jpeg']:
        mp = f"{CONFIG['train_mask_dir']}/{base}{ext}"
        if os.path.exists(mp):
            mask = np.array(Image.open(mp))
            if mask.ndim == 3: mask = mask[:,:,0]
            mask = remap_mask(mask)
            color = mask_to_color(mask)
            axes[row,0].imshow(img);   axes[row,0].set_title('Image'); axes[row,0].axis('off')
            axes[row,1].imshow(color); axes[row,1].set_title('Mask');  axes[row,1].axis('off')
            axes[row,2].imshow((0.5*img+0.5*color).astype(np.uint8)); axes[row,2].set_title('Overlay'); axes[row,2].axis('off')
            break
plt.tight_layout(); plt.show()
# ── STEP 15: FINAL SCORE ───────────────────────────────────
print("\n" + "="*55)
print("🏆 FINAL SCORES")
print("="*55)
print(f'   B2 alone  : {best_iou1:.4f}')
print(f'   B0 alone  : {best_iou2:.4f}')
print(f'   Ensemble  : {ensemble_miou:.4f} ← SUBMIT THIS!')
print("="*55)

# Download results
from google.colab import files
for f in ['confusion_matrix.png','per_class_iou.png','segmentation_viz.png']:
    path = f"{CONFIG['output_dir']}/{f}"
    if os.path.exists(path):
        files.download(path)
        print(f'✅ Downloaded {f}')

print('\n🎉 ALL DONE! Team BuildSquad 🏆')