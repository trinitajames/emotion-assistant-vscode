
"""
Minimal training stub: replace with your Colab training loop.
Goal: produce weights/emotion_cnn.pt compatible with MiniEmotionCNN (48x48 grayscale).
"""
import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import MiniEmotionCNN

class DummyFER(Dataset):
    def __init__(self, n=1000):
        self.n = n
        self.tf = transforms.Compose([transforms.Grayscale(), transforms.Resize((48,48)), transforms.ToTensor()])
    def __len__(self): return self.n
    def __getitem__(self, idx):
        import PIL.Image as Image
        import numpy as np
        img = Image.fromarray((np.random.rand(64,64,3)*255).astype('uint8'))
        x = self.tf(img)  # (1,48,48)
        y = torch.randint(0,7,(1,)).item()
        return x, y

def main():
    os.makedirs("weights", exist_ok=True)
    model = MiniEmotionCNN()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ds = DummyFER(256)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()
    for epoch in range(3):
        model.train()
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            l = loss(out, y)
            l.backward()
            opt.step()
        print(f"epoch {epoch+1} done")
    torch.save(model.state_dict(), "weights/emotion_cnn.pt")
    print("Saved -> weights/emotion_cnn.pt")

if __name__ == "__main__":
    main()
