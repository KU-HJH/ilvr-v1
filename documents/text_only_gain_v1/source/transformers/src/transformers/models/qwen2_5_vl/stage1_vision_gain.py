"""Stage 1 teacher-input latent-body attention intervention only."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Stage1VisionGainRequest:
    batch_index: int
    start_marker: int
    end_marker: int


def build_stage1_gain_attention_mask(padding, request, dtype):
    if padding.ndim != 2 or not bool(((padding==0)|(padding==1)).all()):
        raise ValueError('Stage 1 gain requires binary [B,T] padding')
    if not isinstance(request,Stage1VisionGainRequest) or not dtype.is_floating_point:
        raise TypeError('invalid Stage 1 gain request or attention dtype')
    b,l,r=request.batch_index,request.start_marker,request.end_marker
    if any(type(x) is not int for x in (b,l,r)) or not (0<=b<padding.shape[0] and 0<=l<l+1<r<padding.shape[1]):
        raise ValueError('Stage 1 gain block coordinates are invalid')
    positions=torch.arange(padding.shape[1],device=padding.device)
    q,k=positions[:,None],positions[None,:]
    allowed=((k<=q)[None,None]&padding[:,None,None,:].bool()).clone()
    allowed[b,0]&=~((q>=r)&(k>l)&(k<r))
    return torch.zeros(allowed.shape,device=padding.device,dtype=dtype).masked_fill(~allowed,-torch.inf)
