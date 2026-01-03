import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torchvision import datasets, transforms
import os

# argparse
parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--dataset_target', type=str, default='webcam', help='Path to the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--model_target', type=str, default='densenet', help='The target model to test')
parser.add_argument('--attack_method', type=str, default='all', help='The attack method to use')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use (or "cpu" to use CPU)')
parser.add_argument('--model_path', type=str, default=None, help='Path to the model to test')
args = parser.parse_args()


kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


if args.gpu.lower() == 'cpu' or not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{args.gpu}")


# Import ART components and attacks
from art.attacks.evasion import (
    ProjectedGradientDescent,
    FastGradientMethod,
    CarliniL2Method,
    PixelAttack,
    SpatialTransformation
)
from art.estimators.classification import PyTorchClassifier

if "Product" in args.dataset_target or "RealWorld" in args.dataset_target:
    num_classes = 65
    data_path = f"./datasets/office_home/{args.dataset_target}"
    model_path = f"DSAN/models/target_{str(args.dataset_target).lower()}_{args.model_target}.pt"
    root_path_save = f"defense/models/office_home"
else:
    num_classes = 31
    data_path = f"./datasets/office31/{args.dataset_target}"
    model_path = f"DSAN/models/target_{str(args.dataset_target).lower()}_{args.model_target}.pt"
    root_path_save = f"defense/models/office31"

if args.model_path is not None:
    model_path = args.model_path

print(num_classes)

def load_data(input_size):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x])
                      for x in ['train', 'test']}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, **kwargs)
                    for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    return data_loaders, dataset_sizes

def data_loader_to_numpy(data_loader):
    data = []
    labels = []
    for inputs, labels_batch in data_loader:
        data.append(inputs.numpy())
        labels.append(labels_batch.numpy())

    return np.concatenate(data), np.concatenate(labels)



input_size = 224
data_loaders, dataset_sizes = load_data(input_size)


model = torch.load(model_path, weights_only=False, map_location=device)
model.to(device)
model.eval()  # Set in evaluation mode
#print(f"Model output classes: {model.classifier.out_features}")
print(f"Dataset target : {args.dataset_target}")
print(f"Model target : {args.model_target}")
print(f"Model path : {model_path}")
print(f"Data path : {data_path}")

input_shape = (3, input_size, input_size)  # e.g., for a color image of size 32x32

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create the ART classifier wrapper
classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=input_shape,
    nb_classes=num_classes,
    clip_values=(0.0, 1.0)
)
# -------------------------------
x_test, y_test = data_loader_to_numpy(data_loaders['test'])


def generate_adversarial_examples(classifier, x_test, y_test, method):
    # (a) Fast Gradient Sign Method (FGSM)
    if method == 'fgsm' or method == 'all':
        torch.cuda.empty_cache()
        print(f"Beginning of FGSM attack on {args.dataset_target} dataset with {args.model_target} model")
        attack_fgsm = FastGradientMethod(estimator=classifier, eps=0.1, batch_size=args.batch_size)
        x_adv_fgsm = attack_fgsm.generate(x=x_test)
        print("FGSM adversarial example shape:", x_adv_fgsm.shape)
        if method=='fgsm':
            return x_adv_fgsm


    if method == 'pgd' or method == 'all':
        torch.cuda.empty_cache()
        print(f"Beginning of PGD attack on {args.dataset_target} dataset with {args.model_target} model")
        # (b) Projected Gradient Descent (PGD)
        attack_pgd = ProjectedGradientDescent(estimator=classifier, eps=0.1, max_iter=40, batch_size=args.batch_size)
        x_adv_pgd = attack_pgd.generate(x=x_test)
        predict_accuracy(classifier, x_adv_pgd)
        if method == 'pgd':
            return x_adv_pgd

    if method == 'cw' or method == 'all':
        # (e) Carlini & Wagner (C&W) Attack (using L2)
        torch.cuda.empty_cache()
        print(f"Beginning of CW attack on {args.dataset_target} dataset with {args.model_target} model")
        attack_cw = CarliniL2Method(classifier=classifier, max_iter=10, confidence=0.0, batch_size=args.batch_size, verbose=True)
        x_adv_cw = attack_cw.generate(x=x_test, y=y_test)
        predict_accuracy(classifier, x_adv_cw)
        if method == 'cw':
            return x_adv_cw

    if method == 'pa' or method == 'all':
        # (f) Pixel Attack
        torch.cuda.empty_cache()
        print(f"Beginning of Pixel attack on {args.dataset_target} dataset with {args.model_target} model")
        pixel_attack = PixelAttack(classifier=classifier, max_iter=100, targeted=False, verbose=True)
        x_adv_patch = pixel_attack.generate(x=x_test, y=y_test)
        predict_accuracy(classifier, x_adv_patch)
        if method == 'pa':
            return x_adv_patch

    if method == 'st' or method == 'all':
        torch.cuda.empty_cache()
        print(f"Beginning of ST attack on {args.dataset_target} dataset with {args.model_target} model")
        # (g) Spatial Transformation (ST) Attack
        attack_st = SpatialTransformation(classifier=classifier, max_translation=5, verbose=True)
        x_adv_st = attack_st.generate(x=x_test, y=y_test)
        predict_accuracy(classifier, x_adv_st)
        if method == 'st':
            return x_adv_st
    if method == 'all':
        return x_adv_fgsm, x_adv_pgd, x_adv_cw, x_adv_patch, x_adv_st
    else:
        raise ValueError(f"Unknown attack method: {method}")

def predict_accuracy(classifier, x_adv):
    predictions = classifier.predict(x_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
    print("Accuracy on adversarial examples: {:.2f}%".format(accuracy * 100))

if args.attack_method == 'all':
    x_fgsm, x_pgd, x_cw, x_pa, x_st = generate_adversarial_examples(classifier=classifier, x_test=x_test, y_test=y_test, method=args.attack_method)
    predict_accuracy(classifier, x_fgsm)
    predict_accuracy(classifier, x_pgd)
    predict_accuracy(classifier, x_cw)
    predict_accuracy(classifier, x_pa)
    predict_accuracy(classifier, x_st)
else:
    x = generate_adversarial_examples(classifier=classifier, x_test=x_test, y_test=y_test, method=args.attack_method)
    predict_accuracy(classifier, x)