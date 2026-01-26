import jittor as jt
import jittor.nn as nn
import jittor.models as jt_models
from collections import OrderedDict
import math
import os


class PseudoCombiner(nn.Module):
    def __init__(self, no_classes, pretrained=False, backbone_name="resnet18"):
        super(PseudoCombiner, self).__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = self.create_backbone(
            backbone_name, pretrained, no_classes
        )
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, no_classes)

    def execute(self, x):
        outputs = []
        pseudo_no = len(x)

        x = jt.concat(x, dim=0)
        x = self.backbone(x)
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], -1))
        splits = jt.split(x, x.shape[0] // pseudo_no, dim=0)

        for pseudo in splits:
            outputs.append(self.classifier(pseudo))

        if (not self.is_training()) and len(outputs) > 1:
            outputs.append(
                jt.pow(nn.softmax(outputs[0], dim=1), 0.25)
                * jt.pow(nn.softmax(outputs[1], dim=1), 0.75)
            )
            outputs.append(
                jt.pow(nn.softmax(outputs[0], dim=1), 0.50)
                * jt.pow(nn.softmax(outputs[1], dim=1), 0.50)
            )
            outputs.append(
                jt.pow(nn.softmax(outputs[0], dim=1), 0.75)
                * jt.pow(nn.softmax(outputs[1], dim=1), 0.25)
            )

        return outputs

    def create_backbone(self, backbone_name, pretrained, no_classes):
        if backbone_name == "resnet18":
            backbone = jt_models.resnet18(pretrained=pretrained)
            feature_dim = 512
            if hasattr(backbone, "fc"):
                backbone.fc = nn.Identity()
        elif backbone_name == "vit_small":
            backbone = ViTSmall(pretrained=pretrained)
            feature_dim = backbone.ft_dim
        elif backbone_name.lower() == "caffenet":
            backbone = AlexNetCaffe()
            for m in backbone.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, 0.1)
                    nn.init.constant_(m.bias, 0.0)
            if pretrained:
                state_dict = jt.load("./Pretrained_Models/alexnet_caffe_jittor.pkl")
                backbone.load_state_dict(state_dict)
            # Keep classifier as Sequential; slicing returns list, so rewrap it.
            backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])
            feature_dim = 4096
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        return backbone, feature_dim


class LocalResponseNorm(nn.Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=2.0):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def execute(self, x):
        n, c, h, w = x.shape
        pad = self.size // 2
        sq = x * x
        padded = jt.zeros((n, c + 2 * pad, h, w))
        padded[:, pad : pad + c, :, :] = sq
        sums = []
        for i in range(c):
            sums.append(padded[:, i : i + self.size, :, :].sum(dim=1))
        scale = self.k + (self.alpha / self.size) * jt.stack(sums, dim=1)
        return x / jt.pow(scale, self.beta)


class ViTSmall(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        pretrained=False,
    ):
        super(ViTSmall, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = jt.zeros((1, 1, embed_dim))
        self.pos_embed = jt.zeros((1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ft_dim = embed_dim

        self.apply(self._init_weights)

        weight_path = "./Pretrained_Models/vit_small_jittor.pkl"
        if pretrained:
            if os.path.isfile(weight_path):
                try:
                    state_dict = jt.load(weight_path)
                    self.load_state_dict(state_dict)
                    print("Loaded ViT-Small pretrained weights from Pretrained_Models/vit_small_jittor.pkl")
                except Exception as exc:
                    print(f"Warning: failed to load ViT-Small weights ({exc}), using random init.")
            else:
                print("Warning: ViT-Small pretrained weights file not found; using random init.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Jittor lacks torch-style normal_, so use randn and scale
            with jt.no_grad():
                m.weight.assign(jt.randn(m.weight.shape) * 0.02)
                if m.bias is not None:
                    m.bias.assign(jt.zeros_like(m.bias))
        elif isinstance(m, nn.LayerNorm):
            with jt.no_grad():
                m.bias.assign(jt.zeros_like(m.bias))
                m.weight.assign(jt.ones_like(m.weight))

    def execute(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.broadcast((B, 1, self.ft_dim))
        x = jt.concat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def execute(self, x):
        x = self.proj(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.permute((0, 2, 1))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, attn_drop=0.0, proj_drop=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape((B, N, 3, self.num_heads, C // self.num_heads))
        qkv = qkv.permute((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = jt.bmm(q, k.transpose((0, 1, 3, 2))) * self.scale
        attn = nn.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = jt.bmm(attn, v)
        x = x.transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def execute(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AlexNetCaffe(nn.Module):
    def __init__(self, dropout=True):
        super(AlexNetCaffe, self).__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv(3, 96, 11, stride=4)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.Pool(3, stride=2, ceil_mode=True)),
                    ("norm1", LocalResponseNorm(5, 1.0e-4, 0.75)),
                    ("conv2", nn.Conv(96, 256, 5, padding=2, groups=2)),
                    ("relu2", nn.ReLU()),
                    ("pool2", nn.Pool(3, stride=2, ceil_mode=True)),
                    ("norm2", LocalResponseNorm(5, 1.0e-4, 0.75)),
                    ("conv3", nn.Conv(256, 384, 3, padding=1)),
                    ("relu3", nn.ReLU()),
                    ("conv4", nn.Conv(384, 384, 3, padding=1, groups=2)),
                    ("relu4", nn.ReLU()),
                    ("conv5", nn.Conv(384, 256, 3, padding=1, groups=2)),
                    ("relu5", nn.ReLU()),
                    ("pool5", nn.Pool(3, stride=2, ceil_mode=True)),
                ]
            )
        )
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc6", nn.Linear(256 * 6 * 6, 4096)),
                    ("relu6", nn.ReLU()),
                    ("drop6", nn.Dropout()),
                    ("fc7", nn.Linear(4096, 4096)),
                    ("relu7", nn.ReLU()),
                    ("drop7", nn.Dropout()),
                    ("fc8", nn.Linear(4096, 1000)),
                ]
            )
        )

    def execute(self, x):
        x = self.features(x * 57.6)
        x = x.reshape((x.shape[0], -1))
        x = self.classifier(x)
        return x
