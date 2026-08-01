import os
import re
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

def parse_logs(log_file):
    train_losses = defaultdict(list)
    val_losses = {}
    val_accuracies = {}
    val_ema_accuracies = {}
    
    last_loss = None
    current_epoch = -1
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse train loss
            train_match = re.search(r'Train: \[(\d+)/\d+\]\[\d+/\d+\]\s+.*loss ([\d.]+)', line)
            if train_match:
                current_epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                train_losses[current_epoch].append(loss)
                
            # Parse test loss (validation loss)
            test_match = re.search(r'Test: \[.*?\].*Loss [\d.]+ \(([\d.]+)\)', line)
            if test_match:
                last_loss = float(test_match.group(1))
                
            # Parse val accuracy from summary line
            val_match = re.search(r'\* Acc@1 ([\d.]+)', line)
            if val_match:
                acc = float(val_match.group(1))
                # The first time we see accuracy for this epoch, it is normal validation
                if current_epoch not in val_accuracies:
                    val_accuracies[current_epoch] = acc
                    if last_loss is not None:
                        val_losses[current_epoch] = last_loss
                # The second time, it is EMA validation
                else:
                    val_ema_accuracies[current_epoch] = acc

    # Average train losses per epoch
    avg_train_losses = {epoch: sum(losses)/len(losses) for epoch, losses in train_losses.items()}
    
    return avg_train_losses, val_losses, val_accuracies, val_ema_accuracies

def plot_curves(avg_train_losses, val_losses, val_accuracies, val_ema_accuracies, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Loss Curves
    ax1 = axs[0, 0]
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    if avg_train_losses:
        t_epochs = sorted(avg_train_losses.keys())
        t_losses = [avg_train_losses[e] for e in t_epochs]
        ax1.plot(t_epochs, t_losses, 'b.-', label='Training Loss')
    
    if val_losses:
        v_epochs = sorted(val_losses.keys())
        v_loss_vals = [val_losses[e] for e in v_epochs]
        ax1.plot(v_epochs, v_loss_vals, 'r.-', label='Validation Loss')
        
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Top-1 Validation Accuracy
    ax2 = axs[0, 1]
    ax2.set_title('Top-1 Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    if val_accuracies:
        v_epochs = sorted(val_accuracies.keys())
        v_accs = [val_accuracies[e] for e in v_epochs]
        ax2.plot(v_epochs, v_accs, 'r.-', label='Validation Acc@1')
        
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Best Accuracy Progress
    ax3 = axs[1, 1]
    ax3.set_title('Best Accuracy Progress')
    ax3.set_xlabel('Checkpoint')
    ax3.set_ylabel('Best Accuracy (%)')
    
    best_accs = []
    current_best = 0
    if val_accuracies:
        for e in sorted(val_accuracies.keys()):
            current_best = max(current_best, val_accuracies[e])
            best_accs.append(current_best)
        ax3.plot(range(len(best_accs)), best_accs, 'b.-', label='Best Acc@1')
        
    best_ema_accs = []
    current_ema_best = 0
    if val_ema_accuracies:
        for e in sorted(val_ema_accuracies.keys()):
            current_ema_best = max(current_ema_best, val_ema_accuracies[e])
            best_ema_accs.append(current_ema_best)
        ax3.plot(range(len(best_ema_accs)), best_ema_accs, 'g.-', label='EMA Best Acc@1')
        
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Hide the 4th subplot
    axs[1, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training curves from log file.')
    parser.add_argument('--log', type=str, default='output/tiny/finetune/log_rank0.txt', help='Path to the log file')
    parser.add_argument('--out', type=str, default='training_curves.png', help='Path to save the output plot')
    args = parser.parse_args()
    
    if not os.path.exists(args.log):
        print(f"Error: Log file {args.log} not found.")
        exit(1)
            
    avg_train_losses, val_losses, val_accuracies, val_ema_accuracies = parse_logs(args.log)
    
    if not avg_train_losses and not val_accuracies:
        print("Warning: No data parsed from logs. Please check the log format.")
        
    plot_curves(avg_train_losses, val_losses, val_accuracies, val_ema_accuracies, args.out)
