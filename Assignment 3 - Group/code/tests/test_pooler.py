from __future__ import annotations
import pytest
import torch
from video3d.config import AdapterConfig
from video3d.models.encoders import token_coverage
from video3d.models.pooler import MaskPooler


def test_foreground_preserves_temporal_spatial_order() -> None:
    masks=torch.zeros(1,4,16,32)
    masks[:,:2,:,:16]=1
    masks[:,2:,:,16:]=1
    assert (token_coverage(masks)>=0.5).tolist()==[[True,False,False,True]]


def test_pooler_excludes_background_and_retains_native_non_rgb() -> None:
    model=MaskPooler(AdapterConfig(d_vid=32,d_cond=16,heads=4,pooler_queries=4))
    tokens=torch.randn(2,4,32,requires_grad=True)
    coverage=torch.tensor([[1.,0,0,0],[1.,1,1,1]])
    cond={'image':torch.randn(2,4,16),'mask':torch.randn(2,4,16),'pointmap':torch.randn(2,8,16)}
    output=model.condition(cond,tokens,coverage,['image'])
    assert output['image'].shape==(2,4,16)
    assert torch.equal(output['mask'],cond['mask'])
    assert torch.equal(output['pointmap'],cond['pointmap'])
    changed=tokens.detach().clone()
    changed[0,1:]+=1000
    assert torch.allclose(output['image'],model.condition(cond,changed,coverage,['image'])['image'])
    output['image'].square().mean().backward()
    assert model.pooler.query_tokens.grad.abs().sum()>0
    assert tokens.grad[0,0].abs().sum()>0
    assert tokens.grad[0,1:].abs().sum()==0
    with pytest.raises(ValueError,match='No foreground'):
        model.condition(cond,tokens,torch.zeros_like(coverage),['image'])


def test_pooler_bfloat16_backward_is_finite() -> None:
    model=MaskPooler(AdapterConfig(d_vid=32,d_cond=16,heads=4,pooler_queries=4))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=model.condition({'image':torch.randn(1,4,16)},torch.randn(1,8,32),torch.ones(1,8),['image'])['image']
        output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(model.pooler.query_tokens.grad).all()
