"""Phase-dominant bounded FIR with exactly matched-EVM noise/global controls."""
import math
import torch
from s17b_core import HERE
from s17_augment import bounded_kernel,as_complex,as_iq

def apply(x,tau,generator,auxiliary,mode='physical',probability=1.,details=False):
    n=len(x);device=x.device;selected=torch.rand(n,device=device,generator=generator)<probability
    c=(torch.rand(n,2,device=device,generator=generator)*2-1)*(.15*tau)
    u=torch.linspace(-1,1,16,device=device)[None]
    angle=c[:,:1]*torch.cos(math.pi*u)+c[:,1:]*torch.sin(2*math.pi*u)
    h=bounded_kernel(n,device,phase=angle);z=as_complex(x)
    transformed=torch.fft.ifft(torch.fft.fft(z,n=512)*torch.fft.fft(h,n=512))[:,8:264]
    power=z.abs().square().mean(1,keepdim=True).clamp_min(1e-12)
    energy=(transformed-z).abs().square().mean(1,keepdim=True)
    evm=(energy/power).sqrt()
    if mode=='physical':out=transformed
    elif mode=='noise':
        noise=as_complex(torch.randn(x.shape,device=device,dtype=x.dtype,generator=auxiliary))
        out=z+noise*(energy/noise.abs().square().mean(1,keepdim=True).clamp_min(1e-12)).sqrt()
    elif mode=='global':
        sign=torch.where(torch.rand(n,1,device=device,generator=auxiliary)<.5,-1.,1.)
        rotation=sign*2*torch.asin((evm/2).clamp(max=1));out=z*torch.exp(1j*rotation)
    else:raise ValueError('Unregistered intervention mode')
    out=torch.where(selected[:,None,None],as_iq(out),x)
    if details:
        spectrum=torch.fft.fft(h,n=512).abs().clamp_min(1e-9)
        return out,dict(evm=evm[:,0],amplitude_ripple_db=(20*torch.log10(spectrum)).abs().amax(1))
    return out
