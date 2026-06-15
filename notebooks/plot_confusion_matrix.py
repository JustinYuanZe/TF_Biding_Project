"""
Auto-refactored script: plot_confusion_matrix.py
Refactored to align with project standards.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import FileLink, display, Image

# 1. Safe inference function to retrieve targets and predictions
def get_predictions_and_targets(model, loader, device="cuda"):
    """
    Safely runs evaluation on the loader to retrieve true labels,
    predicted labels, and class probabilities. Handles PyTorch tensors
    on GPU/CPU robustly to prevent TypeErrors.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            shape_features = batch["shape_features"].to(device)
            labels = batch["labels"]
            
            logits = model(input_ids, attention_mask, shape_features)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            # Safely convert labels to CPU numpy arrays
            if isinstance(labels, torch.Tensor):
                all_targets.extend(labels.cpu().numpy())
            else:
                all_targets.extend(labels)
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

# 2. English plotting function for both raw and normalized confusion matrices
def plot_and_save_confusion_matrix(targets, preds, class_names, save_path="confusion_matrix.png"):
    """
    Generates and saves a two-panel English confusion matrix plot:
    - Left panel: Absolute sample counts.
    - Right panel: Normalized percentages (recall per class).
    """
    cm = confusion_matrix(targets, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, data, title, fmt in [
        (ax1, cm, "Confusion Matrix (Counts)", "d"),
        (ax2, cm_norm, "Confusion Matrix (Normalized)", ".2%"),
    ]:
        im = ax.imshow(data, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        
        # Colorbar configuration
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add labels to each cell
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = data[i, j]
                ax.text(
                    j, i, format(val, fmt),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=12, fontweight="bold"
                )
                
    plt.suptitle("G-CMAB Model Confusion Matrix", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    # Create target directory if it doesn't exist
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot successfully saved to: {save_path}")

# 3. Main execution block
if __name__ == "__main__":
    # Check if required environment variables are available
    try:
        # Detect active PyTorch device
        device = next(model.parameters()).device
        print(f"Running inference on device: {device}")
        
        # Step 1: Run inference to extract predictions and targets
        print("Extracting targets and predictions from test_loader...")
        all_targets, all_preds, all_probs = get_predictions_and_targets(model, test_loader, device=device)
        print("Successfully extracted predictions!")
        
        # Step 2: Print English classification report
        class_names = cfg.CLASS_NAMES if 'cfg' in globals() else ['SP1', 'SP2', 'SP4', 'Negative']
        print("\n" + "=" * 60)
        print(" ENGLISH CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
        
        # Step 3: Plot and save confusion matrix
        save_file = "confusion_matrix.png"
        plot_and_save_confusion_matrix(all_targets, all_preds, class_names, save_path=save_file)
        
        # Step 4: Display the image in the Kaggle/Jupyter notebook
        display(Image(filename=save_file))
        
        # Step 5: Generate a clickable download link in Kaggle
        print("\n" + "=" * 60)
        print(" DOWNLOAD LINK")
        print("=" * 60)
        print("Click the link below to download the confusion matrix plot directly:")
        display(FileLink(save_file))
        
    except NameError as ne:
        print(f"NameError: {ne}")
        print("Please ensure that 'model', 'test_loader', and 'cfg' are loaded and active in your notebook scope.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
