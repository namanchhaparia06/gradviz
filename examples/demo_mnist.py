import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from gradviz import GradViz

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = M().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train = datasets.MNIST(root="./data", train=True, download=True,
                           transform=transforms.ToTensor())
    loader = DataLoader(train, batch_size=128, shuffle=True)

    gv = GradViz(model)
    gv.attach()
    for epoch in range(2):
        gv.set_epoch(epoch)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            gv.step()
        print(f"epoch {epoch} done")
    gv.detach()
    gv.save("gradviz_logs/mnist.csv")
    gv.plot(by="layer", topk=10, show=False, savepath="gradviz_logs/lines.png")
    gv.heatmap(by="layer", show=False, savepath="gradviz_logs/heatmap.png")

if __name__ == "__main__":
    main()
