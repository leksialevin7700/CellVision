import os
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from scripts.dataset import CancerDataset
from scripts.utils import calculate_metrics, save_model
def main(
    data_dir='./data',
    save_path='best_model.pth',
    num_epochs=10,
    batch_size=8,
    lr=1e-3,
    val_split=0.2,
    seed=42
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['benign', 'malignant']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = CancerDataset(data_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.flatten())
                all_probs.extend(probs.flatten())

        metrics = calculate_metrics(all_labels, all_preds, all_probs, classes=classes)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {metrics['accuracy']:.2%} | F1: {metrics['f1']:.2f}")

        if metrics['f1'] > best_f1:
            print("Saving new best model...")
            save_model(model, save_path)
            best_f1 = metrics['f1']

    print("Training complete.")
    print("Best F1:", best_f1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        save_path=args.save_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
    )