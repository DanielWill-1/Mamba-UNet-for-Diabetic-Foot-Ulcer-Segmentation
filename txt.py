"""

Mamba-UNet++ v4 — DFUC2022 Edition (Targeting ~0.73+ SOTA Dice)

================================================================

Key changes from generic v4:

  1.  TTA averages raw logits (removed unstable reverse-sigmoid)

  2.  DataLoader num_workers=4 with proper thread guards

  3.  HD95 removed from epoch validation; only computed at inference

  4.  LovaszSoftmaxLoss → LovaszHingeLoss (binary task)

  5.  Patch-based training (512×512 crops centred on ulcers)

  6.  Sliding-window inference at native resolution

  7.  Aspect-ratio-preserving pad-to-square preprocessing

  8.  Loss schedule front-loads Tversky (FN-heavy) for extreme

      class imbalance (89 % of ulcers < 5 % image area)

"""



# ── Thread guards MUST come before any cv2 / numpy import ────────────

import os



os.environ["OMP_NUM_THREADS"] = "1"

os.environ["MKL_NUM_THREADS"] = "1"

os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["NUMEXPR_NUM_THREADS"] = "1"



import math, random, warnings

import cv2



cv2.setNumThreads(0)

cv2.ocl.setUseOpenCL(False)



import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import albumentations as A

from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from scipy.ndimage import distance_transform_edt

from copy import deepcopy



warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True



# ── optional fast Mamba ──────────────────────────────────────────────

try:

    from mamba_ssm import Mamba



    USE_FAST_MAMBA = True

    print("✓ Fast CUDA Mamba loaded")

except ImportError:

    USE_FAST_MAMBA = False

    print("⚠  mamba-ssm not found — using CBAM attention (fast fallback)")



# ── optional timm ────────────────────────────────────────────────────

try:

    import timm



    HAS_TIMM = True

    print("✓ timm loaded — will use pretrained EfficientNet-B4 encoder")

except ImportError:

    HAS_TIMM = False

    print("⚠  timm not found — falling back to scratch encoder")



# ── DFUC2022 constants ───────────────────────────────────────────────

TRAIN_PATCH_SIZE = 512  # crop size for training

INFER_PATCH_SIZE = 512  # sliding-window tile at inference

INFER_STRIDE = 384  # overlap = 128 px

PAD_RESIZE = 640  # pad-to-square then resize to this for val

IMG_SIZE = TRAIN_PATCH_SIZE  # network input resolution

image_dir = 'datasets/DFUC2022_train_release/DFUC2022_train_images'

mask_dir = 'datasets/DFUC2022_train_release/DFUC2022_train_masks'

def set_seed(s=42):

    random.seed(s)

    np.random.seed(s)

    torch.manual_seed(s)

    torch.cuda.manual_seed_all(s)





set_seed()





# =====================================================================

# 1. HELPERS — Aspect-ratio-preserving pad-to-square

# =====================================================================

def pad_to_square(image, mask=None, pad_value=0):

    """Pad the shorter side so the image becomes square."""

    h, w = image.shape[:2]

    if h == w:

        if mask is not None:

            return image, mask, (0, 0, 0, 0)

        return image, (0, 0, 0, 0)



    diff = abs(h - w)

    pad_before = diff // 2

    pad_after = diff - pad_before



    if h > w:

        # pad width

        top, bottom, left, right = 0, 0, pad_before, pad_after

    else:

        # pad height

        top, bottom, left, right = pad_before, pad_after, 0, 0



    image = cv2.copyMakeBorder(

        image, top, bottom, left, right,

        cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value),

    )

    if mask is not None:

        mask = cv2.copyMakeBorder(

            mask, top, bottom, left, right,

            cv2.BORDER_CONSTANT, value=0,

        )

        return image, mask, (top, bottom, left, right)

    return image, (top, bottom, left, right)





def unpad(arr, pad_info, orig_h, orig_w):

    """Remove padding applied by pad_to_square."""

    top, bottom, left, right = pad_info

    h, w = arr.shape[:2]

    arr = arr[

        top : h - bottom if bottom else h,

        left : w - right if right else w,

    ]

    if arr.shape[0] != orig_h or arr.shape[1] != orig_w:

        arr = cv2.resize(

            arr.astype(np.float32), (orig_w, orig_h),

            interpolation=cv2.INTER_LINEAR,

        )

    return arr





# =====================================================================

# 2. DATASET — Patch-based for DFUC2022

# =====================================================================

class DFUC2022Dataset(Dataset):

    """

    Training mode  : extracts random 512×512 crops that contain

                     the ulcer (if one exists).  Falls back to a

                     random crop if there is no foreground.

    Validation mode: pads to square → resizes to PAD_RESIZE.

    """



    def __init__(

        self,

        image_dir,

        mask_dir,

        names,

        transform=None,

        training=True,

        patch_size=TRAIN_PATCH_SIZE,

        crops_per_image=1,

    ):

        self.transform = transform

        self.training = training

        self.patch_size = patch_size

        self.crops_per_image = crops_per_image

        self.samples = []

        for n in names:

            ip = os.path.join(image_dir, n)

            if not os.path.exists(ip) or os.path.getsize(ip) == 0:

                continue

            stem = n.rsplit(".", 1)[0]

            mp_png = os.path.join(mask_dir, stem + ".png")

            mp_jpg = os.path.join(mask_dir, n)

            mp = (

                mp_png

                if os.path.exists(mp_png)

                else (mp_jpg if os.path.exists(mp_jpg) else None)

            )

            self.samples.append({"img": ip, "mask": mp})



    def __len__(self):

        if self.training:

            return len(self.samples) * self.crops_per_image

        return len(self.samples)



    # ── crop helpers ─────────────────────────────────────────────────

    @staticmethod

    def _bbox_of_mask(mask):

        """Return (y_min, y_max, x_min, x_max) of foreground."""

        ys, xs = np.where(mask > 0.5)

        if len(ys) == 0:

            return None

        return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())



    def _random_crop_around_fg(self, image, mask):

        """

        Extract a patch_size × patch_size crop guaranteed to

        overlap the foreground bounding box.  If the image is

        smaller than patch_size on either axis the image is

        padded first.

        """

        ps = self.patch_size

        h, w = image.shape[:2]



        # pad if image smaller than patch

        pad_h = max(ps - h, 0)

        pad_w = max(ps - w, 0)

        if pad_h > 0 or pad_w > 0:

            image = cv2.copyMakeBorder(

                image, 0, pad_h, 0, pad_w,

                cv2.BORDER_CONSTANT, value=(0, 0, 0),

            )

            mask = cv2.copyMakeBorder(

                mask, 0, pad_h, 0, pad_w,

                cv2.BORDER_CONSTANT, value=0,

            )

            h, w = image.shape[:2]



        bbox = self._bbox_of_mask(mask)

        if bbox is not None:

            y0, y1, x0, x1 = bbox

            # ensure crop overlaps bbox

            cy_lo = max(0, y1 - ps + 1)

            cy_hi = min(h - ps, y0)

            cx_lo = max(0, x1 - ps + 1)

            cx_hi = min(w - ps, x0)

            cy = random.randint(

                min(cy_lo, cy_hi), max(cy_lo, cy_hi)

            )

            cx = random.randint(

                min(cx_lo, cx_hi), max(cx_lo, cx_hi)

            )

        else:

            cy = random.randint(0, h - ps)

            cx = random.randint(0, w - ps)



        return image[cy : cy + ps, cx : cx + ps], mask[cy : cy + ps, cx : cx + ps]



    def __getitem__(self, idx):

        real_idx = idx % len(self.samples)

        s = self.samples[real_idx]



        im = cv2.imread(s["img"])

        im = (

            cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            if im is not None

            else np.zeros((TRAIN_PATCH_SIZE, TRAIN_PATCH_SIZE, 3), np.uint8)

        )

        mk = (

            cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)

            if s["mask"]

            else np.zeros(im.shape[:2], np.uint8)

        )

        mk = (mk > 127).astype(np.float32)



        if self.training:

            # ── patch-based crop ─────────────────────────────────

            im, mk = self._random_crop_around_fg(im, mk)

        else:

            # ── pad to square → resize ───────────────────────────

            im, mk, _ = pad_to_square(im, mk, pad_value=0)

            im = cv2.resize(im, (PAD_RESIZE, PAD_RESIZE))

            mk = cv2.resize(

                mk, (PAD_RESIZE, PAD_RESIZE),

                interpolation=cv2.INTER_NEAREST,

            )



        if self.transform:

            aug = self.transform(image=im, mask=mk)

            im, mk = aug["image"], aug["mask"].unsqueeze(0)

        return im, mk





def get_train_transforms():

    return A.Compose(

        [

            # patch is already TRAIN_PATCH_SIZE, just augment

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

            A.Affine(

                scale=(0.85, 1.15),

                translate_percent=(-0.08, 0.08),

                rotate=(-30, 30),

                p=0.5,

            ),

            A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.2),

            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),

            A.RandomBrightnessContrast(0.2, 0.2, p=0.4),

            A.HueSaturationValue(8, 20, 10, p=0.3),

            A.GaussNoise(var_limit=(5, 20), p=0.15),

            A.Normalize(

                mean=(0.485, 0.456, 0.406),

                std=(0.229, 0.224, 0.225),

            ),

            ToTensorV2(),

        ]

    )





def get_val_transforms():

    return A.Compose(

        [

            # image is already PAD_RESIZE × PAD_RESIZE

            A.Normalize(

                mean=(0.485, 0.456, 0.406),

                std=(0.229, 0.224, 0.225),

            ),

            ToTensorV2(),

        ]

    )





# =====================================================================

# 3. METRICS (no HD95 — CPU bottleneck removed from epoch loop)

# =====================================================================

def calc_metrics(pred_logits, target, thr=0.5):

    pred = (torch.sigmoid(pred_logits) > thr).float().cpu().numpy()

    tgt = target.float().cpu().numpy()

    dice_l, iou_l, prec_l, rec_l = [], [], [], []

    for p, t in zip(pred, tgt):

        p, t = p.squeeze(), t.squeeze()

        tp = float((p * t).sum())

        fp = float((p * (1 - t)).sum())

        fn = float(((1 - p) * t).sum())

        dice_l.append((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6))

        iou_l.append((tp + 1e-6) / (tp + fp + fn + 1e-6))

        prec_l.append((tp + 1e-6) / (tp + fp + 1e-6))

        rec_l.append((tp + 1e-6) / (tp + fn + 1e-6))

    return map(np.mean, [dice_l, iou_l, prec_l, rec_l])





def hausdorff_95(pred_logits, target, thr=0.5):

    """Expensive CPU metric — only called at final inference."""

    pred = (torch.sigmoid(pred_logits) > thr).float().cpu().numpy()

    tgt = target.float().cpu().numpy()

    hd_l = []

    for p, t in zip(pred, tgt):

        p, t = p.squeeze().astype(bool), t.squeeze().astype(bool)

        if not p.any() or not t.any():

            hd_l.append(float(np.sqrt(p.shape[0] ** 2 + p.shape[1] ** 2)))

            continue

        d1 = distance_transform_edt(~p)[t]

        d2 = distance_transform_edt(~t)[p]

        hd_l.append(np.percentile(np.concatenate([d1, d2]), 95))

    return np.mean(hd_l)





# =====================================================================

# 4. LOSSES

# =====================================================================

class DiceBCELoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss()



    def forward(self, logits, targets):

        logits = logits.float()

        targets = targets.float()

        bce = self.bce(logits, targets)

        p = torch.sigmoid(logits).view(-1)

        t = targets.view(-1)

        inter = (p * t).sum()

        dice = 1 - (2 * inter + self.smooth) / (

            p.sum() + t.sum() + self.smooth

        )

        return bce, dice





class LovaszHingeLoss(nn.Module):

    """Binary Lovász-Hinge loss (renamed from LovaszSoftmaxLoss)."""



    def forward(self, logits, targets):

        logits = logits.float()

        targets = targets.float()

        probs = torch.sigmoid(logits).view(-1)

        tgts = targets.view(-1)

        if tgts.sum() == 0 or tgts.sum() == tgts.numel():

            return (logits * 0.0).mean()

        errors = (tgts - probs).abs()

        errors_sorted, perm = torch.sort(errors, descending=True)

        gt_sorted = tgts[perm]

        grad = self._lovasz_grad(gt_sorted)

        return torch.clamp(torch.dot(errors_sorted, grad), 0.0, 2.0)



    @staticmethod

    def _lovasz_grad(gt_sorted):

        gts = gt_sorted.sum()

        intersection = gts - gt_sorted.cumsum(0)

        union = gts + (1 - gt_sorted).cumsum(0)

        jaccard = 1.0 - intersection / (union + 1e-6)

        if gt_sorted.numel() > 1:

            jaccard[1:] = jaccard[1:] - jaccard[:-1]

        return jaccard





class TverskyLoss(nn.Module):

    """alpha > beta penalises FN more — critical for DFUC2022."""



    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-3):

        super().__init__()

        self.alpha = alpha   # FP weight (lower = tolerate more FP)

        self.beta = beta     # FN weight (higher via 1-alpha trick)

        self.gamma = gamma   # focal exponent

        self.smooth = smooth



    def forward(self, inputs, targets):

        p = torch.sigmoid(inputs.float()).view(-1)

        t = targets.float().view(-1)

        tp = (p * t).sum()

        fp = (p * (1 - t)).sum()

        fn = ((1 - p) * t).sum()

        tv = (tp + self.smooth) / (

            tp + self.alpha * fp + self.beta * fn + self.smooth

        )

        return (1 - tv) ** self.gamma





class ScheduledLoss(nn.Module):

    """

    DFUC2022 schedule: front-load Tversky (FN-penalising) and

    DiceBCE to combat extreme foreground sparsity, then shift

    towards Lovász for boundary refinement.



    Weights order: (BCE, Dice, Tversky, Lovász)

    """



    def __init__(self):

        super().__init__()

        self.dice_bce = DiceBCELoss()

        self.tversky = TverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)

        self.lovasz = LovaszHingeLoss()

        self._epoch = 0



    def set_epoch(self, e):

        self._epoch = e



    def forward(self, logits, targets):

        bce, dice = self.dice_bce(logits, targets)

        tv = self.tversky(logits, targets)

        lov = self.lovasz(logits, targets)



        def safe(t):

            return (

                t

                if torch.isfinite(t)

                else torch.zeros(1, device=logits.device)[0]

            )



        e = self._epoch

        #                    BCE   Dice  Tversky  Lovász

        if e < 20:

            w = (0.25, 0.25, 0.40, 0.10)  # recall-heavy

        elif e < 50:

            w = (0.20, 0.30, 0.30, 0.20)  # balanced

        elif e < 90:

            w = (0.15, 0.30, 0.25, 0.30)  # boundary refine

        else:

            w = (0.10, 0.30, 0.20, 0.40)  # late Lovász



        return (

            w[0] * safe(bce)

            + w[1] * safe(dice)

            + w[2] * safe(tv)

            + w[3] * safe(lov)

        )





# =====================================================================

# 5. BUILDING BLOCKS

# =====================================================================

class SEBlock(nn.Module):

    def __init__(self, c, r=8):

        super().__init__()

        self.se = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(c, c // r, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(c // r, c, bias=False),

            nn.Sigmoid(),

        )



    def forward(self, x):

        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)





class CBAM(nn.Module):

    def __init__(self, c, r=8):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(c, c // r, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(c // r, c, bias=False),

        )

        self.spatial = nn.Sequential(

            nn.Conv2d(2, 1, 7, padding=3, bias=False),

            nn.BatchNorm2d(1),

            nn.Sigmoid(),

        )



    def forward(self, x):

        avg = x.mean(dim=(2, 3))

        mx = x.amax(dim=(2, 3))

        ca = (

            torch.sigmoid(self.mlp(avg) + self.mlp(mx))

            .unsqueeze(-1)

            .unsqueeze(-1)

        )

        x = x * ca

        sa = torch.cat(

            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)],

            dim=1,

        )

        x = x * self.spatial(sa)

        return x





class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c, drop=0.1):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),

            nn.BatchNorm2d(out_c),

            nn.GELU(),

            nn.Dropout2d(drop),

            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),

            nn.BatchNorm2d(out_c),

            nn.GELU(),

        )

        self.se = SEBlock(out_c)

        self.proj = (

            nn.Conv2d(in_c, out_c, 1, bias=False)

            if in_c != out_c

            else nn.Identity()

        )



    def forward(self, x):

        return self.se(self.conv(x)) + self.proj(x)





class ASPP(nn.Module):

    def __init__(self, in_c, out_c=256):

        super().__init__()

        rates = [1, 6, 12, 18]

        self.convs = nn.ModuleList(

            [

                nn.Sequential(

                    nn.Conv2d(

                        in_c, out_c,

                        3 if r > 1 else 1,

                        padding=r if r > 1 else 0,

                        dilation=r,

                        bias=False,

                    ),

                    nn.BatchNorm2d(out_c),

                    nn.ReLU(inplace=True),

                )

                for r in rates

            ]

        )

        self.pool = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(in_c, out_c, 1, bias=False),

            nn.BatchNorm2d(out_c),

            nn.ReLU(inplace=True),

        )

        self.proj = nn.Sequential(

            nn.Conv2d(out_c * (len(rates) + 1), out_c, 1, bias=False),

            nn.BatchNorm2d(out_c),

            nn.ReLU(inplace=True),

            nn.Dropout2d(0.1),

        )



    def forward(self, x):

        h, w = x.shape[2:]

        feats = [c(x) for c in self.convs]

        feats.append(

            F.interpolate(

                self.pool(x), size=(h, w),

                mode="bilinear", align_corners=True,

            )

        )

        return self.proj(torch.cat(feats, 1))





class AttentionGate(nn.Module):

    def __init__(self, Fg, Fl, Fi):

        super().__init__()

        self.Wg = nn.Sequential(

            nn.Conv2d(Fg, Fi, 1, bias=False), nn.BatchNorm2d(Fi)

        )

        self.Wx = nn.Sequential(

            nn.Conv2d(Fl, Fi, 1, bias=False), nn.BatchNorm2d(Fi)

        )

        self.psi = nn.Sequential(

            nn.Conv2d(Fi, 1, 1, bias=False),

            nn.BatchNorm2d(1),

            nn.Sigmoid(),

        )



    def forward(self, g, x):

        g1 = self.Wg(

            F.interpolate(

                g, x.shape[2:], mode="bilinear", align_corners=True

            )

        )

        out = self.psi(F.relu(g1 + self.Wx(x), inplace=True))

        return x * out





class FastMambaBlock(nn.Module):

    def __init__(self, d):

        super().__init__()

        self.mamba = Mamba(d_model=d, d_state=16, d_conv=4, expand=2)

        self.norm = nn.LayerNorm(d)



    def _scan(self, x, flip=False):

        s = torch.flip(x, [1]) if flip else x

        o = self.mamba(self.norm(s))

        return torch.flip(o, [1]) if flip else o



    def forward(self, x):

        B, C, H, W = x.shape

        h = x.view(B, C, -1).transpose(1, 2)

        oh = (self._scan(h) + self._scan(h, flip=True)) / 2

        v = (

            x.permute(0, 1, 3, 2)

            .contiguous()

            .view(B, C, -1)

            .transpose(1, 2)

        )

        ov = (self._scan(v) + self._scan(v, flip=True)) / 2

        out = (oh + ov) / 2

        return out.transpose(1, 2).view(B, C, H, W) + x





# =====================================================================

# 6. ENCODER

# =====================================================================

class ResidualAdapter(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.needs_proj = in_c != out_c

        if self.needs_proj:

            self.proj = nn.Conv2d(in_c, out_c, 1, bias=False)

            self.bn = nn.BatchNorm2d(out_c)

            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")



    def forward(self, x):

        if not self.needs_proj:

            return x

        return self.bn(self.proj(x))





class EfficientNetEncoder(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        self.stages = timm.create_model(

            "efficientnet_b4",

            pretrained=pretrained,

            features_only=True,

        )

        self.channels = self.stages.feature_info.channels()

        print(f"  EfficientNet-B4 encoder channels: {self.channels}")



    def forward(self, x):

        return self.stages(x)





class ScratchEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        F_ = [32, 64, 128, 256, 512]

        self.channels = F_

        self.s0 = ConvBlock(3, F_[0])

        self.s1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(F_[0], F_[1]))

        self.s2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(F_[1], F_[2]))

        self.s3 = nn.Sequential(

            nn.MaxPool2d(2), ConvBlock(F_[2], F_[3]), CBAM(F_[3])

        )

        self.s4 = nn.Sequential(

            nn.MaxPool2d(2), ConvBlock(F_[3], F_[4]), CBAM(F_[4])

        )



    def forward(self, x):

        f0 = self.s0(x)

        f1 = self.s1(f0)

        f2 = self.s2(f1)

        f3 = self.s3(f2)

        f4 = self.s4(f3)

        return [f0, f1, f2, f3, f4]





# =====================================================================

# 7. MAMBA UNET++ V4 — DFUC2022 EDITION

# =====================================================================

class MambaUNetPPv4(nn.Module):

    def __init__(

        self, num_classes=1, deep_supervision=True, pretrained=True

    ):

        super().__init__()

        self.deep_supervision = deep_supervision



        ContextBlock = FastMambaBlock if USE_FAST_MAMBA else CBAM



        if HAS_TIMM:

            self.encoder = EfficientNetEncoder(pretrained=pretrained)

        else:

            self.encoder = ScratchEncoder()



        enc_ch = self.encoder.channels

        dec_ch = [64, 128, 192, 256, 256]



        self.adapt = nn.ModuleList(

            [ResidualAdapter(ec, dc) for ec, dc in zip(enc_ch, dec_ch)]

        )



        self.aspp = ASPP(dec_ch[4], dec_ch[4])

        self.ctx3 = ContextBlock(dec_ch[3])

        self.ctx4 = ContextBlock(dec_ch[4])



        C = dec_ch

        self.up = nn.Upsample(

            scale_factor=2, mode="bilinear", align_corners=True

        )



        self.ag = nn.ModuleDict(

            {

                "01": AttentionGate(C[1], C[0], C[0] // 2),

                "11": AttentionGate(C[2], C[1], C[1] // 2),

                "21": AttentionGate(C[3], C[2], C[2] // 2),

                "31": AttentionGate(C[4], C[3], C[3] // 2),

                "02": AttentionGate(C[1], C[0], C[0] // 2),

                "12": AttentionGate(C[2], C[1], C[1] // 2),

                "22": AttentionGate(C[3], C[2], C[2] // 2),

                "03": AttentionGate(C[1], C[0], C[0] // 2),

                "13": AttentionGate(C[2], C[1], C[1] // 2),

                "04": AttentionGate(C[1], C[0], C[0] // 2),

            }

        )



        self.d01 = ConvBlock(C[0] + C[1], C[0])

        self.d11 = ConvBlock(C[1] + C[2], C[1])

        self.d21 = ConvBlock(C[2] + C[3], C[2])

        self.d31 = ConvBlock(C[3] + C[4], C[3])

        self.d02 = ConvBlock(C[0] * 2 + C[1], C[0])

        self.d12 = ConvBlock(C[1] * 2 + C[2], C[1])

        self.d22 = ConvBlock(C[2] * 2 + C[3], C[2])

        self.d03 = ConvBlock(C[0] * 3 + C[1], C[0])

        self.d13 = ConvBlock(C[1] * 3 + C[2], C[1])

        self.d04 = ConvBlock(C[0] * 4 + C[1], C[0])



        if deep_supervision:

            self.heads = nn.ModuleList(

                [nn.Conv2d(C[0], num_classes, 1) for _ in range(4)]

            )

        else:

            self.head = nn.Conv2d(C[0], num_classes, 1)



        self._init_decoder_weights()



    def _init_decoder_weights(self):

        for m in self.modules():

            if any(m is s for s in self.encoder.modules()):

                continue

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(

                    m.weight, mode="fan_out", nonlinearity="relu"

                )

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.ones_(m.weight)

                nn.init.zeros_(m.bias)



    def forward(self, x):

        feats = self.encoder(x)

        f = [ad(fe) for ad, fe in zip(self.adapt, feats)]

        f[3] = self.ctx3(f[3])

        f[4] = self.ctx4(self.aspp(f[4]))



        x01 = self.d01(

            torch.cat([self.ag["01"](f[1], f[0]), self.up(f[1])], 1)

        )

        x11 = self.d11(

            torch.cat([self.ag["11"](f[2], f[1]), self.up(f[2])], 1)

        )

        x21 = self.d21(

            torch.cat([self.ag["21"](f[3], f[2]), self.up(f[3])], 1)

        )

        x31 = self.d31(

            torch.cat([self.ag["31"](f[4], f[3]), self.up(f[4])], 1)

        )



        x02 = self.d02(

            torch.cat(

                [f[0], self.ag["02"](x11, x01), self.up(x11)], 1

            )

        )

        x12 = self.d12(

            torch.cat(

                [f[1], self.ag["12"](x21, x11), self.up(x21)], 1

            )

        )

        x22 = self.d22(

            torch.cat(

                [f[2], self.ag["22"](x31, x21), self.up(x31)], 1

            )

        )



        x03 = self.d03(

            torch.cat(

                [f[0], x01, self.ag["03"](x12, x02), self.up(x12)], 1

            )

        )

        x13 = self.d13(

            torch.cat(

                [f[1], x11, self.ag["13"](x22, x12), self.up(x22)], 1

            )

        )

        x04 = self.d04(

            torch.cat(

                [f[0], x01, x02, self.ag["04"](x13, x03), self.up(x13)],

                1,

            )

        )



        H, W = x.shape[2:]

        if self.deep_supervision:

            return [

                F.interpolate(

                    h(o), size=(H, W),

                    mode="bilinear", align_corners=True,

                )

                for h, o in zip(self.heads, [x01, x02, x03, x04])

            ]

        out = self.head(x04)

        return F.interpolate(

            out, size=(H, W), mode="bilinear", align_corners=True

        )





# =====================================================================

# 8. EMA

# =====================================================================

class EMA:

    def __init__(self, model, decay=0.9995):

        self.decay = decay

        self.shadow = deepcopy(model).eval()

        for p in self.shadow.parameters():

            p.requires_grad_(False)



    @torch.no_grad()

    def update(self, model):

        for s, m in zip(

            self.shadow.parameters(), model.parameters()

        ):

            s.copy_(self.decay * s + (1 - self.decay) * m.data)

        for s, m in zip(self.shadow.buffers(), model.buffers()):

            s.copy_(m)



    def __call__(self, *args, **kwargs):

        return self.shadow(*args, **kwargs)





# =====================================================================

# 9. TTA — Fixed: average raw logits, not probabilities

# =====================================================================

@torch.no_grad()

def tta_predict(model, imgs, device):

    """

    Test-Time Augmentation that averages RAW LOGITS across

    four geometric flips, then applies sigmoid once.

    Removes the numerically unstable reverse-sigmoid entirely.

    """

    logit_sum = torch.zeros_like(imgs[:, :1, :, :])  # (B,1,H,W)

    count = 0



    for fh in [False, True]:

        for fv in [False, True]:

            x = imgs.clone()

            if fh:

                x = torch.flip(x, [3])

            if fv:

                x = torch.flip(x, [2])



            with torch.amp.autocast("cuda"):

                out = model(x)

                logits = out[-1] if isinstance(out, list) else out



            # undo geometric transforms on the logit map

            if fv:

                logits = torch.flip(logits, [2])

            if fh:

                logits = torch.flip(logits, [3])



            logit_sum = logit_sum + logits.float()

            count += 1



    return logit_sum / count  # return averaged LOGITS





# =====================================================================

# 10. SLIDING-WINDOW INFERENCE

# =====================================================================

@torch.no_grad()

def sliding_window_inference(

    model, image_np, device,

    patch_size=INFER_PATCH_SIZE,

    stride=INFER_STRIDE,

    use_tta=False,

):

    """

    Run the model on overlapping patches of the (padded-to-square,

    full-resolution) image and stitch logits together.



    Args:

        image_np: RGB uint8 (H, W, 3) — already padded to square.

        Returns: logit map (H, W) float32 numpy.

    """

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)

    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)



    H, W = image_np.shape[:2]

    logit_acc = np.zeros((1, 1, H, W), dtype=np.float32)

    count_acc = np.zeros((1, 1, H, W), dtype=np.float32)



    # pad so last patch fits

    pad_h = (stride - (H - patch_size) % stride) % stride

    pad_w = (stride - (W - patch_size) % stride) % stride

    img_padded = cv2.copyMakeBorder(

        image_np, 0, pad_h, 0, pad_w,

        cv2.BORDER_CONSTANT, value=(0, 0, 0),

    )

    Hp, Wp = img_padded.shape[:2]



    for y in range(0, Hp - patch_size + 1, stride):

        for x in range(0, Wp - patch_size + 1, stride):

            patch = img_padded[y : y + patch_size, x : x + patch_size]

            patch_t = (

                torch.from_numpy(

                    ((patch.astype(np.float32) / 255.0 - mean) / std)

                    .transpose(2, 0, 1)

                )

                .unsqueeze(0)

                .to(device)

            )



            with torch.amp.autocast("cuda"):

                if use_tta:

                    logits = tta_predict(model, patch_t, device)

                else:

                    out = model(patch_t)

                    logits = out[-1] if isinstance(out, list) else out



            logits_np = logits.float().cpu().numpy()  # (1,1,ps,ps)



            # accumulate into the unpadded region only

            y_end = min(y + patch_size, H)

            x_end = min(x + patch_size, W)

            ph = y_end - y

            pw = x_end - x

            logit_acc[:, :, y:y_end, x:x_end] += logits_np[

                :, :, :ph, :pw

            ]

            count_acc[:, :, y:y_end, x:x_end] += 1.0



    count_acc = np.maximum(count_acc, 1.0)

    return (logit_acc / count_acc)[0, 0]  # (H, W)





# =====================================================================

# 11. PLOTTING

# =====================================================================

def plot_all(history, path="training_metrics_v4_dfuc.png"):

    eps = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    fig.suptitle(

        "Mamba-UNet++ v4 DFUC2022 Training Dashboard",

        fontsize=16, fontweight="bold",

    )



    def _p(a, key, label, color, target=None):

        a.plot(

            eps, history[key],

            color=color, marker="o", markersize=2, label=label,

        )

        if target:

            a.axhline(

                target, color="limegreen", ls="--", lw=1,

                alpha=0.7, label=f"Target {target}",

            )

        a.legend(fontsize=8)

        a.grid(True, alpha=0.3)



    _p(ax[0, 0], "train_loss", "Train Loss", "red")

    ax[0, 0].set_title("Loss")

    _p(ax[0, 1], "val_dice", "Val Dice", "royalblue", 0.73)

    ax[0, 1].set_title("Dice")

    ax[0, 1].set_ylim(0, 1)

    _p(ax[0, 2], "val_iou", "Val IoU", "seagreen")

    ax[0, 2].set_title("IoU")

    ax[0, 2].set_ylim(0, 1)



    ax[1, 0].plot(

        eps, history["val_precision"],

        color="purple", marker="o", markersize=2, label="Precision",

    )

    ax[1, 0].plot(

        eps, history["val_recall"],

        color="teal", marker="s", markersize=2, label="Recall",

    )

    ax[1, 0].legend(fontsize=8)

    ax[1, 0].grid(True, alpha=0.3)

    ax[1, 0].set_ylim(0, 1)

    ax[1, 0].set_title("Precision & Recall")



    ax[1, 1].axis("off")

    ax[1, 1].text(

        0.5, 0.5,

        "HD95 computed\nonly at inference\n(removed from\nepoch loop)",

        ha="center", va="center", fontsize=12,

        style="italic", color="gray",

    )



    ax[1, 2].plot(eps, history["lr"], "k", marker="o", markersize=2)

    ax[1, 2].set_title("Learning Rate")

    ax[1, 2].set_yscale("log")

    ax[1, 2].grid(True, alpha=0.3)



    plt.tight_layout()

    plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()





# =====================================================================

# 12. INFERENCE VISUALISATION — Sliding-window + HD95

# =====================================================================

def run_inference(

    model_path, image_dir, mask_dir, num_samples=4, device=None

):

    if device is None:

        device = torch.device(

            "cuda" if torch.cuda.is_available() else "cpu"

        )

    model = MambaUNetPPv4(

        deep_supervision=False, pretrained=False

    ).to(device)

    sd = torch.load(model_path, map_location=device)

    model.load_state_dict(sd, strict=False)

    model.eval()

    print(f"\n✓ Loaded checkpoint from '{model_path}'")



    ext = (".png", ".jpg", ".jpeg")

    imgs = sorted(

        [f for f in os.listdir(image_dir) if f.lower().endswith(ext)]

    )

    picks = random.sample(imgs, min(num_samples, len(imgs)))



    fig, axes = plt.subplots(

        num_samples, 4, figsize=(17, 4 * num_samples)

    )

    if num_samples == 1:

        axes = axes[np.newaxis]

    fig.suptitle(

        "Inference — Mamba-UNet++ v4 DFUC2022 (Sliding Window)",

        fontsize=14, fontweight="bold",

    )

    for c, t in enumerate(

        ["Input", "Ground Truth", "Prediction", "Overlay (TP/FP/FN)"]

    ):

        axes[0, c].set_title(t, fontsize=11, fontweight="bold")



    all_hd95 = []



    for row, name in enumerate(picks):

        ip = os.path.join(image_dir, name)

        mp = os.path.join(mask_dir, name.rsplit(".", 1)[0] + ".png")

        if not os.path.exists(mp):

            mp = os.path.join(mask_dir, name)



        orig = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)

        orig_h, orig_w = orig.shape[:2]



        gt_raw = (

            cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

            if os.path.exists(mp)

            else np.zeros((orig_h, orig_w), np.uint8)

        )

        gt = (gt_raw > 127).astype(np.float32)



        # pad to square for sliding window

        orig_sq, gt_sq, pad_info = pad_to_square(orig, gt, pad_value=0)

        sq_h, sq_w = orig_sq.shape[:2]



        # sliding-window logits

        logit_map = sliding_window_inference(

            model, orig_sq, device,

            patch_size=INFER_PATCH_SIZE,

            stride=INFER_STRIDE,

            use_tta=True,

        )

        pred_sq = (1 / (1 + np.exp(-logit_map)) > 0.5).astype(np.float32)



        # unpad back to original shape

        pred = unpad(pred_sq, pad_info, orig_h, orig_w)

        pred = (pred > 0.5).astype(np.float32)



        # compute metrics on original resolution

        gtt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

        pt = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)

        dice = float(

            (2 * (pt * gtt).sum() + 1e-6) / (pt.sum() + gtt.sum() + 1e-6)

        )

        iou = float(

            ((pt * gtt).sum() + 1e-6)

            / ((pt + gtt - (pt * gtt)).sum() + 1e-6)

        )



        # HD95 — only at inference time

        logit_t = torch.from_numpy(logit_map).unsqueeze(0).unsqueeze(0)

        gt_rsz = cv2.resize(

            gt, (sq_w, sq_h), interpolation=cv2.INTER_NEAREST

        )

        gt_t = torch.from_numpy(gt_rsz).unsqueeze(0).unsqueeze(0)

        hd = hausdorff_95(logit_t, gt_t)

        all_hd95.append(hd)



        # overlay

        disp_size = (512, 512)

        orig_disp = cv2.resize(orig, disp_size)

        gt_disp = cv2.resize(

            gt, disp_size, interpolation=cv2.INTER_NEAREST

        )

        pred_disp = cv2.resize(

            pred, disp_size, interpolation=cv2.INTER_NEAREST

        )



        ov = orig_disp.copy().astype(np.float32)

        tp = (pred_disp > 0.5) & (gt_disp > 0.5)

        fp = (pred_disp > 0.5) & (gt_disp <= 0.5)

        fn = (pred_disp <= 0.5) & (gt_disp > 0.5)

        ov[tp] = ov[tp] * 0.45 + np.array([0, 220, 0]) * 0.55

        ov[fp] = ov[fp] * 0.45 + np.array([220, 0, 0]) * 0.55

        ov[fn] = ov[fn] * 0.45 + np.array([0, 0, 220]) * 0.55



        axes[row, 0].imshow(orig_disp)

        axes[row, 0].set_ylabel(name[:18], fontsize=7)

        axes[row, 1].imshow(gt_disp, cmap="gray")

        axes[row, 2].imshow(pred_disp, cmap="gray")

        axes[row, 2].set_xlabel(

            f"Dice={dice:.4f}  IoU={iou:.4f}  HD95={hd:.1f}px",

            fontsize=8,

        )

        axes[row, 3].imshow(np.clip(ov, 0, 255).astype(np.uint8))

        for ax in axes[row]:

            ax.axis("off")



    patches = [

        mpatches.Patch(color="lime", label="TP"),

        mpatches.Patch(color="red", label="FP"),

        mpatches.Patch(color="blue", label="FN"),

    ]

    fig.legend(

        handles=patches, loc="lower center", ncol=3,

        fontsize=10, bbox_to_anchor=(0.5, -0.005),

    )

    plt.tight_layout()

    plt.savefig(

        "inference_results_v4_dfuc.png", dpi=150, bbox_inches="tight"

    )

    plt.close()

    print("✓ Inference grid saved to 'inference_results_v4_dfuc.png'")

    print(f"  Mean HD95 across samples: {np.mean(all_hd95):.1f} px")





# =====================================================================

# 13. TRAINING — DFUC2022 Edition

# =====================================================================

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(f"GPU   : {torch.cuda.get_device_name(0)}")



    base = "dataset/train"

    img_dir = os.path.join(base, "images")

    msk_dir = os.path.join(base, "labels")

    if not os.path.exists(img_dir):

        img_dir = msk_dir = base



    ext = (".png", ".jpg", ".jpeg")

    all_ = sorted(

        [f for f in os.listdir(img_dir) if f.lower().endswith(ext)]

    )

    if not all_:

        raise ValueError(f"No images in {img_dir}")



    random.shuffle(all_)

    split = int(0.8 * len(all_))

    tr_names = all_[:split]

    vl_names = all_[split:]

    print(f"Train : {len(tr_names)} | Val: {len(vl_names)}")

    print(

        f"Strategy: {TRAIN_PATCH_SIZE}×{TRAIN_PATCH_SIZE} "

        f"ulcer-centred crops (train) | "

        f"pad-to-square + {PAD_RESIZE}px (val)\n"

    )



    tr_ds = DFUC2022Dataset(

        img_dir, msk_dir, tr_names,

        transform=get_train_transforms(),

        training=True,

        patch_size=TRAIN_PATCH_SIZE,

        crops_per_image=2,  # 2 crops per image per epoch

    )

    vl_ds = DFUC2022Dataset(

        img_dir, msk_dir, vl_names,

        transform=get_val_transforms(),

        training=False,

    )



    BATCH = 8  # larger patches → smaller batch

    tr_ld = DataLoader(

        tr_ds, batch_size=BATCH, shuffle=True,

        num_workers=4, pin_memory=True, persistent_workers=True,

    )

    vl_ld = DataLoader(

        vl_ds, batch_size=4, shuffle=False,

        num_workers=4, pin_memory=True, persistent_workers=True,

    )



    model = MambaUNetPPv4(

        deep_supervision=True, pretrained=HAS_TIMM

    ).to(device)



    print(

        "⏭  Skipping torch.compile (avoids pthread exhaustion "

        "in constrained environments)"

    )



    ema = EMA(model, decay=0.9995)

    crit = ScheduledLoss()



    # Differential LR: encoder 10× slower than decoder

    enc_params = list(model.encoder.parameters())

    enc_param_ids = {id(p) for p in enc_params}

    dec_params = [

        p for p in model.parameters() if id(p) not in enc_param_ids

    ]



    optimizer = torch.optim.AdamW(

        [

            {"params": enc_params, "lr": 3e-5, "name": "encoder"},

            {"params": dec_params, "lr": 2e-4, "name": "decoder"},

        ],

        weight_decay=1e-4,

        betas=(0.9, 0.999),

    )



    epochs = 150

    sched = torch.optim.lr_scheduler.OneCycleLR(

        optimizer,

        max_lr=[3e-4, 2e-3],

        epochs=epochs,

        steps_per_epoch=len(tr_ld),

        pct_start=0.05,

        anneal_strategy="cos",

        div_factor=10,

        final_div_factor=100,

    )



    scaler = torch.amp.GradScaler("cuda")



    ds_w = [0.05, 0.15, 0.30, 0.50]

    ACCUM = 2



    best_dice = 0.0

    patience = 0

    PATIENCE = 30

    history = {

        k: []

        for k in [

            "train_loss",

            "val_dice",

            "val_iou",

            "val_precision",

            "val_recall",

            "lr",

        ]

    }



    print(f"Training: {epochs} epochs | OneCycleLR | AdamW | EMA\n")



    for epoch in range(epochs):

        model.train()

        crit.set_epoch(epoch)

        tloss = 0.0

        n_valid = 0

        optimizer.zero_grad(set_to_none=True)



        pbar = tqdm(

            enumerate(tr_ld), total=len(tr_ld),

            desc=f"E{epoch+1:03d}/{epochs} [Train]", ncols=110,

        )



        for step, (imgs, masks) in pbar:

            imgs = imgs.to(device, non_blocking=True)

            masks = masks.to(device, non_blocking=True)



            with torch.amp.autocast("cuda"):

                seg_outs = model(imgs)

                if isinstance(seg_outs, tuple):

                    seg_outs = seg_outs[0]

                if not isinstance(seg_outs, list):

                    seg_outs = [seg_outs]



                loss = (

                    sum(

                        w * crit(o, masks)

                        for w, o in zip(ds_w[-len(seg_outs) :], seg_outs)

                    )

                    / ACCUM

                )



            if not torch.isfinite(loss):

                optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix({"loss": "nan-skip"})

                continue



            scaler.scale(loss).backward()

            n_valid += 1



            if (step + 1) % ACCUM == 0 or (step + 1) == len(tr_ld):

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad(set_to_none=True)

                ema.update(model)



            sched.step()



            loss_val = loss.item() * ACCUM

            tloss += loss_val

            pbar.set_postfix({"loss": f"{loss_val:.4f}"})



        epoch_loss = tloss / max(n_valid, 1)



        # ── Validate with EMA (no HD95 — GPU stays hot) ──────────

        ema.shadow.eval()

        d_l, i_l, p_l, r_l = [], [], [], []

        use_tta = (epoch + 1) >= 80 and (epoch + 1) % 5 == 0



        with torch.no_grad():

            for imgs, masks in tqdm(

                vl_ld,

                desc=f"E{epoch+1:03d}/{epochs} [Val]  ",

                leave=False, ncols=110,

            ):

                imgs = imgs.to(device, non_blocking=True)

                masks = masks.to(device, non_blocking=True)



                if use_tta:

                    final_out = tta_predict(ema.shadow, imgs, device)

                else:

                    with torch.amp.autocast("cuda"):

                        out = ema.shadow(imgs)

                        if isinstance(out, tuple):

                            out = out[0]

                        final_out = (

                            out[-1] if isinstance(out, list) else out

                        )



                d, i, p, r = calc_metrics(final_out, masks)

                d_l.append(d)

                i_l.append(i)

                p_l.append(p)

                r_l.append(r)



        vd, vi, vp, vr = (

            np.mean(d_l), np.mean(i_l), np.mean(p_l), np.mean(r_l)

        )

        cur_lr = optimizer.param_groups[0]["lr"]



        history["train_loss"].append(epoch_loss)

        history["val_dice"].append(vd)

        history["val_iou"].append(vi)

        history["val_precision"].append(vp)

        history["val_recall"].append(vr)

        history["lr"].append(cur_lr)



        tta_tag = " [TTA]" if use_tta else ""

        print(

            f"  Loss:{epoch_loss:.4f} | Dice:{vd:.4f} | "

            f"IoU:{vi:.4f} | Prec:{vp:.4f} | Rec:{vr:.4f} | "

            f"LR:{cur_lr:.1e}{tta_tag}"

        )



        if vd > best_dice:

            best_dice = vd

            torch.save(

                ema.shadow.state_dict(),

                "best_mamba_unetpp_v4_dfuc.pth",

            )

            print(f"  ✓ Best model saved! (Dice={best_dice:.4f})")

            patience = 0

        else:

            patience += 1

            if patience >= PATIENCE:

                print(

                    f"\nEarly stop at epoch {epoch+1}. "

                    f"Best Dice={best_dice:.4f}"

                )

                break



        if (epoch + 1) % 5 == 0:

            plot_all(history)



    print(f"\n{'='*65}")

    print(f" Training done | Best EMA Dice: {best_dice:.4f}")

    print(f"{'='*65}")



    run_inference(

        "best_mamba_unetpp_v4_dfuc.pth",

        img_dir, msk_dir,

        num_samples=4, device=device,

    )





if __name__ == "__main__":

    train()