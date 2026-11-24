"""Paper-derived Ma Spec+Siamese for the fixed 8192-sample Shen windows."""

import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from xa.models import ShenRes


class SpecSiamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 32, 7, stride=2)
        self.blocks = nn.Sequential(ShenRes(32, 32), ShenRes(32, 32),
                                    ShenRes(32, 64, True), ShenRes(64, 64))
        self.dense = nn.Linear(13 * 32 * 64, 512)
        self.project = nn.Linear(512, 256)
        self.head = nn.Linear(256, 10)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        x = F.relu(self.stem(F.pad(features, (3, 3, 2, 3))))
        x = F.avg_pool2d(self.blocks(x), 2).permute(0, 2, 3, 1).flatten(1)
        z = self.project(F.relu(self.dense(x)))
        return self.head(z), z


def spectrum(z):
    z = z * z.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
    frames = z.unfold(-1, 128, 64).contiguous()
    bins = torch.fft.fft(frames, dim=-1) / 128
    bins = torch.cat((bins[..., -26:], bins[..., :26]), dim=-1)
    value = bins.abs().clamp_min(1e-6).log().transpose(1,2).unsqueeze(1)
    return value.contiguous(memory_format=torch.channels_last)


def pair_loss(logits, embeddings, labels):
    total = len(labels)
    half = total // 2
    z = F.normalize(embeddings.float(), dim=1)
    similarity = z @ z.T / .05
    anchors = torch.arange(total, device=z.device)
    similarity = similarity.masked_fill(anchors[:,None] == anchors[None,:], -torch.inf)
    return F.cross_entropy(logits.float(), labels) + F.cross_entropy(similarity, (anchors+half)%total)


class Channel:
    """Exponential PDP with five fractional taps and an eight-tone Jakes envelope."""
    def __init__(self, device):
        axis = torch.arange(5, device=device, dtype=torch.float32)
        power = (-axis).exp()
        power /= power.sum()
        center = (axis * power).sum()
        self.delay_units = axis / (((axis-center).square()*power).sum().sqrt())
        self.amplitude = power.sqrt()
        fft_len = 16384
        self.frequency = (2*math.pi*torch.fft.fftfreq(fft_len,device=device))[None,None,:]
        self.directions = torch.cos(torch.arange(8,device=device)*2*math.pi/8)[None,None,:]
        self.ramp = torch.arange(8192,device=device,dtype=torch.float32)/8191

    def __call__(self, z, generator):
        b = len(z)
        options = dict(device=z.device, generator=generator)
        spread = .005 + .295 * torch.rand((b,1), **options)
        doppler = 5 * torch.rand((b,1,1), **options)
        snr = 10 + 30 * torch.rand((b,1), **options)
        phi = 2 * math.pi * torch.rand((b,5,8), **options)
        edge = 2 * math.pi * .008191 * doppler * self.directions
        gain0 = torch.polar(torch.ones_like(phi), phi).mean(-1) * self.amplitude
        gain1 = torch.polar(torch.ones_like(phi), phi+edge).mean(-1) * self.amplitude
        rotation = -self.frequency * (spread*self.delay_units)[:,:,None]
        shift = torch.polar(torch.ones_like(rotation), rotation)
        transfer0 = (shift * gain0[:,:,None]).sum(1)
        transfer1 = (shift * gain1[:,:,None]).sum(1)
        signal = torch.fft.fft(z, n=16384)
        first = torch.fft.ifft(signal*transfer0)[:,:8192]
        last = torch.fft.ifft(signal*transfer1)[:,:8192]
        channel = first + self.ramp*(last-first)
        noise_scale = (.5 * channel.abs().square().mean(1,keepdim=True)*10**(-snr/10)).sqrt()
        noise = torch.randn((*z.shape,2), **options)
        return channel + noise_scale * torch.view_as_complex(noise)


def synthetic_check():
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(42)
    model = SpecSiamese().to(device, memory_format=torch.channels_last)
    counts = sum(p.numel() for p in model.parameters())
    assert round(counts/1e6,2) == 13.94, counts
    x = torch.view_as_complex(torch.randn((64,8192,2),device=device,generator=generator))
    change = Channel(device)(x,generator)
    features = spectrum(change)
    assert features.shape == (64,1,52,127)
    assert torch.isfinite(features).all() and not torch.equal(change,x)
    y = torch.arange(32,device=device).remainder(10).repeat(2)
    with torch.autocast(device_type=device,dtype=torch.bfloat16):
        logits,z = model(features)
    loss = pair_loss(logits,z,y)
    loss.backward()
    assert logits.shape==(64,10) and z.shape==(64,256)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print("synthetic", counts, float(loss))


if __name__ == "__main__":
    synthetic_check()
