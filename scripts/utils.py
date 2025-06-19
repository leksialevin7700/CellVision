
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def calculate_metrics(y_true, y_pred, y_prob=None, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes) if classes else classification_report(y_true, y_pred)
    return {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model