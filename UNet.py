import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.bn1 = nn.BatchNorm2d(out_ch),
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.nn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.nn(x)


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]
            # upsamples x to size of enc_ftrs if sizes do not match
            if enc_ftrs.shape != x.shape:
                x = F.interpolate(x, size=enc_ftrs.shape[2:], mode='bilinear', align_corners=False)
            # enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=[3,64,128,256,512,1024], dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=True, out_sz=(256, 256), masked_aux=False, _eval=False):
        super().__init__()
        self.masked_aux = masked_aux
        enc_chs = [int(x / 2) for x in enc_chs]
        dec_chs = [int(x / 2) for x in dec_chs]
        enc_chs[0] = 3
        enc_chs = tuple(enc_chs)
        dec_chs = tuple(dec_chs)

        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = [nn.Conv2d(dec_chs[-1], num_class, 1)]
        # 32768
        if _eval:
            self.head.append(nn.Softmax(dim=1))
            self.rot_head.append(nn.Softmax(dim=1))
        self.head = nn.Sequential(*self.head)

        if self.masked_aux:
            self.masked_head        = nn.Sequential(*[nn.Conv2d(dec_chs[-1], 16, 1),
                                                    nn.BatchNorm2d(16),
                                                    nn.ReLU(),
                                                    nn.Conv2d(16, 8, 1),
                                                    nn.BatchNorm2d(8),
                                                    nn.ReLU(),
                                                    nn.Conv2d(8, 4, 1),
                                                    nn.BatchNorm2d(4),
                                                    nn.ReLU(),
                                                    nn.Conv2d(4, 3, 1)])
        else: # use rotation aux task
            rot_head_feature_size = 64 * dec_chs[-1] #* out_sz[0] * out_sz[1] // 32 // 32
            self.rot_head    = [nn.Linear(rot_head_feature_size, enc_chs[-1]), nn.ReLU(), nn.Linear(enc_chs[-1], 4)]
            self.rot_head = nn.Sequential(*self.rot_head)

        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs        = self.encoder(x)
        out_latent      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out             = self.head(out_latent)

        if self.masked_aux:
            rot_pred = self.masked_head(out_latent)
            rot_pred = F.interpolate(rot_pred, self.out_sz)
        else:
            # downsample enc_flat to 8 x 8
            out_latent = F.interpolate(out_latent, (8, 8))
            # make rotation prediction from final layer of decoder
            enc_flat = torch.flatten(out_latent, start_dim=1)
            rot_pred = self.rot_head(enc_flat)

        if self.retain_dim:
            #out = F.interpolate(out, self.out_sz, mode='bilinear')
            out = F.interpolate(out, self.out_sz)
        return out, rot_pred
