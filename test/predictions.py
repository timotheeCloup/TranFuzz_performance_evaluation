# load test dataset for models
# load models
# run inference and calculate accuracy : percentage of correct predictions

import torch
import tqdm
import argparse
from torchvision import datasets, transforms
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DSAN.utils.data_loader import load_data

# Ask with argparse for the target name 
parser = argparse.ArgumentParser(description='Generate adversarial samples by improving DeepXplore fuzzing method')
parser.add_argument('--target_model', type=str, default=None, help='target model name')
parser.add_argument('--target_dataset', type=str, default=None, help='target dataset name')
parser.add_argument('--target_path', type=str, default=None, help='robust model training method')
parser.add_argument('--fuzz', default=0, type=bool, help='Test of fuzzed images')
parser.add_argument('--robust', default=None, type=str, help='Name of robust model')
args = parser.parse_args()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the trained model
if "webcam" in args.target_dataset or "amazon" in args.target_dataset:
    trained_model_path = f"DSAN/models/office31/target_{args.target_dataset}_{args.target_model}.pt"
    dataset_test_path = f"datasets/office31/{args.target_dataset}/test/"

if "Product" in args.target_dataset or "RealWorld" in args.target_dataset:
    trained_model_path = f"DSAN/models/office_home/target_{args.target_dataset}_{args.target_model}.pt"
    dataset_test_path = f"datasets/office_home/{args.target_dataset}/test/"

if args.target_path:
    trained_model_path = args.target_path
    

if args.fuzz:
    if args.robust is not None:
        dataset_test_path = f"fuzz/data/target_{args.target_dataset}_{args.target_model}_adv_{args.robust}/"
    else:
        dataset_test_path = f"fuzz/data/target_{args.target_dataset}_{args.target_model}/"

def load_model(model_path):
    model = torch.load(model_path, weights_only=False, map_location=DEVICE)
    # if the model is a PyTorchClassifier from ART, we need to extract the model
    if hasattr(model, "model"):
        model = model.model

    model.eval()
    model = model.to(DEVICE)  # envoie le modèle sur le bon device
    return model


data_loader, class_to_idx, dataset_sizes = load_data(
    data_folder=dataset_test_path,
    batch_size=1,
    train_flag=False,
    kwargs={"num_workers": 0},
)

model = load_model(trained_model_path)
predictions = []
correct_predictions = 0

with tqdm.tqdm(data_loader, desc="Testing") as pbar:
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)  # envoie tes données sur le même device
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            correct_predictions += (predicted == labels).sum().item()

print("Pour le dataset :", dataset_test_path, "avec", args.target_model)
print(f"Accuracy: {correct_predictions / len(predictions):.2%}")
