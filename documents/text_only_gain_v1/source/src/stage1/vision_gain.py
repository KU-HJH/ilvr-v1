"""Stage 1 adjacent-item GT scoring and projected auxiliary directions."""
from dataclasses import dataclass
import torch
from torch import Tensor
from .types import Stage1BatchMetadata


@dataclass(frozen=True)
class Stage1GainBlock:
    source_index: int
    source_id: str | None
    batch_index: int
    block_index: int
    start_marker: int
    end_marker: int
    target_positions: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class Stage1GainPlan:
    blocks: tuple[Stage1GainBlock, ...]
    excluded: tuple[tuple[int, int, str], ...]


def build_stage1_gain_plan(metadata: Stage1BatchMetadata, input_ids: Tensor,
                          labels: Tensor, padding: Tensor) -> Stage1GainPlan:
    if input_ids.ndim != 2 or labels.shape != input_ids.shape or padding.shape != input_ids.shape:
        raise ValueError('gain inputs must share [B,T] shape')
    if len(metadata.samples) != input_ids.shape[0]:
        raise ValueError('gain metadata batch shape mismatch')
    blocks, excluded = [], []
    for row, sample in enumerate(metadata.samples):
        if sample.text_items is None:
            raise ValueError(f'source_index={sample.source_index}: missing text-item metadata')
        structural = set(sample.non_text_positions)
        structural.update(p for span in sample.latent_spans for p in range(span.start,span.end+1))
        for bi, span in enumerate(sample.latent_spans):
            adjacent=[item for item in sample.text_items if item.preceding_block_index == bi]
            if len(adjacent)>1:
                raise ValueError('ambiguous adjacent text-item boundary')
            positions=()
            if adjacent:
                item=adjacent[0]
                if not span.end+1 == item.start <= item.end <= sample.unpadded_length:
                    raise ValueError('invalid adjacent text-item boundary')
                positions=tuple(i for i in range(item.start,item.end)
                    if bool(padding[row,i]) and int(labels[row,i]) != -100 and i not in structural
                    and i not in item.excluded_positions)
            if not positions:
                excluded.append((sample.source_index,bi,'empty_post')); continue
            blocks.append(Stage1GainBlock(sample.source_index,sample.source_id,row,bi,
                span.start,span.end,positions,tuple(int(labels[row,i]) for i in positions)))
    return Stage1GainPlan(tuple(blocks),tuple(excluded))


def score_stage1_gain_block(logits: Tensor, block: Stage1GainBlock) -> Tensor:
    if logits.ndim != 3 or not block.target_positions:
        raise ValueError('gain scoring requires [B,T,V] logits and nonempty GT window')
    rows=logits[block.batch_index,[i-1 for i in block.target_positions]].float()
    ids=torch.tensor(block.target_ids,device=rows.device,dtype=torch.long)
    if bool(((ids<0)|(ids>=rows.shape[-1])).any()):
        raise ValueError('gain target ID outside vocabulary')
    return rows.log_softmax(-1).gather(1,ids[:,None]).mean()


@dataclass(frozen=True)
class Stage1VisionGainConfig:
    alpha: float = 0.0
    min_h_norm_sq: float = 1e-30
    execution: str = 'independent'
    branch_mode: str = 'independent'

    def __post_init__(self):
        import math
        if self.execution not in {'independent', 'packed'}:
            raise ValueError('vision gain execution must be independent or packed')
        if self.branch_mode not in {'independent', 'cumulative'}:
            raise ValueError('vision gain branch mode must be independent or cumulative')
        if self.branch_mode == 'cumulative' and self.execution != 'packed':
            raise ValueError('cumulative vision gain requires packed execution')
        for name,value in (('alpha',self.alpha),('min_h_norm_sq',self.min_h_norm_sq)):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'vision gain {name} must be finite')
        if self.alpha<0 or self.min_h_norm_sq<=0:
            raise ValueError('vision gain alpha must be nonnegative and norm guard positive')


@dataclass(frozen=True)
class ProjectionResult:
    direction: tuple[Tensor, ...]
    projected: bool
    skipped: bool
    zero_direction: bool
    h_dot_g: float
    h_norm_sq: float


def project_gain_direction(g: tuple[Tensor,...], h: tuple[Tensor,...], *,
                           min_h_norm_sq: float) -> ProjectionResult:
    Stage1VisionGainConfig(0.,min_h_norm_sq)
    if len(g)!=len(h) or not g or any(a.shape!=b.shape for a,b in zip(g,h)):
        raise ValueError('projection requires matching nonempty gradient tuples')
    dot=sum((a.double()*b.double()).sum() for a,b in zip(g,h))
    norm=sum(b.double().square().sum() for b in h)
    if not bool(torch.isfinite(dot)&torch.isfinite(norm)) or any(not bool(torch.isfinite(a).all()) for a in g):
        raise FloatingPointError('nonfinite vision gain gradient')
    d,n=dot.item(),norm.item()
    skipped=0<n<min_h_norm_sq
    projected=not skipped and n>0 and d<0
    if skipped:
        direction=tuple(torch.zeros_like(a) for a in g)
    elif projected:
        direction=tuple((a.double()-(d/n)*b.double()).to(a.dtype) for a,b in zip(g,h))
    else:
        direction=tuple(a.clone() for a in g)
    if any(not bool(torch.isfinite(v).all()) for v in direction):
        raise FloatingPointError('nonfinite vision gain projection')
    zero=all(not bool(torch.count_nonzero(v)) for v in direction)
    return ProjectionResult(direction,projected,skipped,zero,d,n)
