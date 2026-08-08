import torch
from utils.augmentation import ChannelAttention, ROTATION_CONSTANT, augment_features, prepare_training_batch, rotation_matrix_y
from utils.knn import build_hierarchy

def test_batch_doubling_and_graph_copy():
    x=torch.randn(2,32,3); anno=x[:,:4]; h=build_hierarchy(x,[4])
    out,a,hd=prepare_training_batch(x,anno,h,True,option=0)
    assert out.shape[0]==4 and a.shape[0]==4
    assert not torch.equal(out[2:,:,:3],x[:,:,:3])
    for k in h[0]: torch.testing.assert_close(hd[0][k][:2],hd[0][k][2:])
    eo,ea,eh=prepare_training_batch(x,anno,h,False)
    assert eo.shape[0]==2 and ea.shape[0]==2 and eh is h

def test_y_rotation_and_constant():
    assert ROTATION_CONSTANT == 3.14592653
    theta=.4; x=torch.tensor([[[1.,2.,3.]]])
    expected=x.reshape(-1,3) @ torch.tensor([[torch.cos(torch.tensor(theta)),0,-torch.sin(torch.tensor(theta))],[0,1,0],[torch.sin(torch.tensor(theta)),0,torch.cos(torch.tensor(theta))]])
    torch.testing.assert_close(augment_features(x,option=1,theta=theta).reshape(-1,3),expected)
    torch.testing.assert_close(rotation_matrix_y(theta), expected.new_tensor([[torch.cos(torch.tensor(theta)),0,-torch.sin(torch.tensor(theta))],[0,1,0],[torch.sin(torch.tensor(theta)),0,torch.cos(torch.tensor(theta))]]))

def test_jitter_shared_across_batch():
    x=torch.zeros(2,10,3); out=augment_features(x,option=2,generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(out[0]-x[0],out[1]-x[1])

def test_channel_attention_shape_and_softmax_axis():
    x = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    attention = ChannelAttention(num_points=4)
    with torch.no_grad(): attention.dense.weight.copy_(torch.tensor([[1., 0., 0., 0.]]))
    out, scores = attention(x)
    assert out.shape == x.shape and scores.shape == (2, 1, 3)
    torch.testing.assert_close(scores.sum(dim=-1), torch.ones(2, 1))
    expected = torch.softmax(x.transpose(1, 2)[..., 0], dim=-1).unsqueeze(1)
    torch.testing.assert_close(scores, expected)

def test_attention_is_applied_only_to_augmented_training_half():
    x=torch.ones(1,4,3); anno=x[:,:2]; h=build_hierarchy(x,[2])
    attention=ChannelAttention(4)
    out,_,_=prepare_training_batch(x,anno,h,True,option=0,channel_attention=attention)
    torch.testing.assert_close(out[:1],x)
    assert not torch.equal(out[1:],x)
