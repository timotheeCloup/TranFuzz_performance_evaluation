import re
import sys
import matplotlib.pyplot as plt

def parse_file(filename):
    """
    Reads the given file and extracts epoch numbers, train loss, train accuracy,
    test loss, and test accuracy.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Regular expressions to match:
    # - Epoch number: "Epoch <n>/..."
    # - Train metrics: "train Loss: <loss> Acc: <acc>%"
    # - Test metrics: "test Loss: <loss> Acc: <acc>%"
    epoch_re = re.compile(r"Epoch\s+(\d+)/")
    train_re = re.compile(r"train Loss:\s*([\d\.]+)\s+Acc:\s*([\d\.]+)%")
    test_re = re.compile(r"test Loss:\s*([\d\.]+)\s+Acc:\s*([\d\.]+)%")
    
    epochs = []
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    current_epoch = None
    for line in lines:
        epoch_match = epoch_re.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)

        train_match = train_re.search(line)
        if train_match and current_epoch is not None:
            train_loss.append(float(train_match.group(1)))
            train_acc.append(float(train_match.group(2)))

        test_match = test_re.search(line)
        if test_match and current_epoch is not None:
            test_loss.append(float(test_match.group(1)))
            test_acc.append(float(test_match.group(2)))
    
    return epochs, train_loss, train_acc, test_loss, test_acc

def trim_data(epochs, train_loss, train_acc, test_loss, test_acc):
    """
    Trims all lists to the same length (the minimum among them)
    to avoid dimension mismatches when plotting.
    """
    min_len = min(len(epochs), len(train_loss), len(train_acc), len(test_loss), len(test_acc))
    return (
        epochs[:min_len],
        train_loss[:min_len],
        train_acc[:min_len],
        test_loss[:min_len],
        test_acc[:min_len]
    )

def plot_metrics(epochs, train_loss, train_acc, test_loss, test_acc, output):
    """
    Creates a figure with two subplots: one for Loss and one for Accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Loss
    ax1.plot(epochs, train_loss, marker='o', label='Train Loss')
    ax1.plot(epochs, test_loss, marker='o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss per Epoch')
    ax1.legend()

    # Plot Accuracy
    ax2.plot(epochs, train_acc, marker='o', label='Train Accuracy')
    ax2.plot(epochs, test_acc, marker='o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy per Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python plot_training.py <log_file.txt> <output_file.png>")
        sys.exit(1)
        
    filename = sys.argv[1]
    output = sys.argv[2]
    epochs, train_loss, train_acc, test_loss, test_acc = parse_file(filename)

    if not epochs:
        print("No epoch data found in the file.")
        sys.exit(1)

    # Ensure that all lists have the same length
    epochs, train_loss, train_acc, test_loss, test_acc = trim_data(epochs, train_loss, train_acc, test_loss, test_acc)
    
    plot_metrics(epochs, train_loss, train_acc, test_loss, test_acc, output)
