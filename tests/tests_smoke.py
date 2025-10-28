import torch
from gradviz import GradViz

def test_collects_gradients():
    m = torch.nn.Linear(4, 3)
    x = torch.randn(5, 4)
    y = torch.randn(5, 3)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    gv = GradViz(m)
    gv.attach()
    for step in range(3):
        opt.zero_grad()
        out = m(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        opt.step()
        gv.step()
    gv.detach()

    df = gv.to_dataframe()
    assert not df.empty
    assert {"step", "param", "layer", "grad_norm"}.issubset(df.columns)
