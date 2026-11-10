"""Bounded 17-tap linear FIR / widely-linear IQ interventions."""
import math
import torch
import torch.nn.functional as F

def as_complex(x):return torch.complex(x[:,0],x[:,1])
def as_iq(z):return torch.stack([z.real,z.imag],1)

def bounded_kernel(n,device,log_db=None,phase=None):
    length=512
    gain=torch.zeros(n,length,device=device) if log_db is None else F.interpolate(log_db[:,None],size=length,mode='linear',align_corners=True)[:,0]
    angle=torch.zeros_like(gain) if phase is None else F.interpolate(phase[:,None],size=length,mode='linear',align_corners=True)[:,0]
    desired=torch.exp(math.log(10)/20*gain+1j*angle)
    impulse=torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(desired,dim=1)),dim=1)[:,248:265]
    impulse=impulse*torch.hann_window(17,periodic=False,device=device)[None]
    impulse=impulse/impulse.abs().square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)
    identity=torch.zeros_like(impulse);identity[:,8]=1
    residual=impulse-identity
    # ||h-delta||_1 <= .25 implies .75 <= |H(f)| <= 1.25 for every frequency.
    impulse=identity+residual*(.25/residual.abs().sum(1,keepdim=True).clamp_min(.25))
    return impulse

def fir(x,log_db=None,phase=None):
    impulse=bounded_kernel(len(x),x.device,log_db,phase)
    # 512 >= 256+17-1: zero-padded linear convolution, then remove known 8-sample delay.
    result=torch.fft.ifft(torch.fft.fft(as_complex(x),n=512)*torch.fft.fft(impulse,n=512))[:,8:264]
    return as_iq(result)

def perturb(x,family,tau,profiles,generator,noise_generator=None,control=False,probability=1.):
    n=len(x);device=x.device
    selected=torch.rand(n,device=device,generator=generator)<probability
    if family=='identity':return x
    if family=='global_phase':return as_iq(as_complex(x)*torch.exp(torch.tensor(1j,device=device)))
    if family=='global_gain':return 2*x
    if family=='phase':
        c=(torch.rand((n,2),device=device,generator=generator)*2-1)*tau*.15
        u=torch.linspace(-1,1,16,device=device)[None];angle=c[:,:1]*torch.cos(math.pi*u)+c[:,1:]*torch.sin(2*math.pi*u)
        out=fir(x,phase=angle)
    else:
        pool=torch.as_tensor(profiles[family],dtype=x.dtype,device=device)
        a=torch.randint(len(pool),(n,),device=device,generator=generator);b=torch.randint(len(pool),(n,),device=device,generator=generator)
        delta=pool[a]-pool[b]
        if family=='amp':out=fir(x,log_db=delta.clamp(-3,3)*tau)
        elif family=='iq':
            beta=torch.complex(delta[:,0],delta[:,1])/2
            beta=beta*(.05/beta.abs().clamp_min(.05))*tau
            z=as_complex(x);out=as_iq((z+beta[:,None]*z.conj())/(1+beta.abs().square()).sqrt()[:,None])
        else:raise ValueError(f'Unregistered perturbation {family}')
    if control:
        z=as_complex(x);target=as_complex(out);energy=(target-z).abs().square().mean(1,keepdim=True)
        noise=torch.randn(x.shape,device=device,dtype=x.dtype,generator=noise_generator);w=as_complex(noise)
        w=w*(energy/w.abs().square().mean(1,keepdim=True).clamp_min(1e-12)).sqrt();out=as_iq(z+w)
    return torch.where(selected[:,None,None],out,x)
