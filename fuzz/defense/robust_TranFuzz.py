import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
import os
import argparse
from torch.utils.data import ConcatDataset, DataLoader
import copy
import tqdm

# =============================================================================
# 0. Set up argument parser
# =============================================================================
parser = argparse.ArgumentParser(description='Adversarial Training with Fuzzed Images')
parser.add_argument('--dataset_target', type=str, default='webcam', help='Path to the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--model_target', type=str, default='densenet', help='The target model')
parser.add_argument('--fuzzed_images_percentage', type=int, default=10, help='Percentage of fuzzed images to use')
parser.add_argument('--clean_training', type=bool, default=False, help='Whether to train on a new model or continue training')
args = parser.parse_args()

# =============================================================================
# Device setup
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    Initialize these variables which will be set in this if statement. Each of these
    variables is model specific.'''
    :param model_name:
    :param num_classes:
    :param feature_extract:
    :param use_pretrained:
    :return:
    """

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# =============================================================================
# Load datasets (clean + fuzzed)
# =============================================================================
def load_datasets(input_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_path, 'test'), test_transform)

    # Load fuzzed dataset if it exists
    fuzzed_dataset = None
    if os.path.exists(fuzzed_images_dir):
        fuzzed_full = datasets.ImageFolder(fuzzed_images_dir, train_transform)
        fuzzed_size = int(len(fuzzed_full) * (args.fuzzed_images_percentage / 100.0))
        fuzzed_dataset, _ = torch.utils.data.random_split(fuzzed_full, [fuzzed_size, len(fuzzed_full) - fuzzed_size])

    if fuzzed_dataset:
        combined_train = ConcatDataset([train_dataset, fuzzed_dataset])
    else:
        combined_train = train_dataset

    train_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    image_datasets = {'train': combined_train, 'test': test_dataset}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
                    for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return data_loaders, dataset_sizes

# =============================================================================
# Training loop
# =============================================================================
def model_training(model, data_loaders, dataset_sizes):
    """
    training target attack model
    :param epoch: num of epoch
    :param model: target model architecture
    :param train_loader: training data loader
    :return:
    """

    best_acc = 0.0
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking

    # Initialize optimizer, criterion, and scaler once
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            print('{0} Phase: {1} {2}'.format('-' * 10, phase, '-' * 10))

            for batch_idx, (inputs, labels) in tqdm.tqdm(enumerate(data_loaders[phase]),
                                                         total=int(dataset_sizes[phase]/args.batch_size),
                                                         desc='Train epoch = {}'.format(epoch),
                                                         ncols=80, leave=False):

                inputs, labels = inputs.to(device), labels.to(device)

                # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(preds)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        # scheduler.step()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, 100 * epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:

                best_acc = epoch_acc
                print('Best val Acc: {:2f}%'.format(100 * best_acc))

                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)  # success or not
                torch.save(model, save_path)

        print()

    return model, save_path


def model_test(model_saved_path, data_loaders):
    print()
    print('starting offline testing')
    print()

    model = torch.load(model_saved_path,weights_only=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in data_loaders['test']:

            images, labels = images.to(device), labels.to(device)
            outputs = model(images).argmax(axis=-1)
            # print(outputs)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()

    print('Accuracy of the network on the test images: {:2f}%'.format(100 * correct / total))

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # =============================================================================
    # Determine dataset details
    # =============================================================================
    input_size = 224
    if "Product" in args.dataset_target or "RealWorld" in args.dataset_target:
        num_classes = 65
        data_path = f"./datasets/office_home/{args.dataset_target}"
        model_path = f"DSAN/models/target_{args.model_target}_{args.dataset_target}_{args.model_target}.pt"
        save_path = f"defense/models/office_home/{args.model_target}_{str(args.dataset_target).lower()}_TranFuzz_defended.pt"
    else:
        num_classes = 31
        data_path = f"./datasets/office31/{args.dataset_target}"
        model_path = f"DSAN/models/target_{args.model_target}_{args.dataset_target}_{args.model_target}.pt"
        save_path = f"defense/models/office31/{args.model_target}_{str(args.dataset_target).lower()}_TranFuzz_defended.pt"

    fuzzed_images_dir = f"fuzz/data/target_{args.dataset_target}_{args.model_target}"

    # =============================================================================
    # Load model, loss & optimizer
    # =============================================================================
    if args.clean_training:
        # Recreate a new target model
        model, input_size = initialize_model(args.model_target, num_classes, feature_extract=True)
        model = model.to(device)
        model.train()
    else: 
        model = torch.load(model_path, map_location=device, weights_only=False)
        model = model.to(device)
        model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # =============================================================================
    # Load datasets
    # =============================================================================
    data_loaders, dataset_sizes = load_datasets(input_size)
    print(f'Data loaded: {dataset_sizes} size')

    # =============================================================================
    # Train model
    # =============================================================================
    model, model_saved_path = model_training(model, data_loaders, dataset_sizes)
    print(f'Model saved at: {model_saved_path}')

    # =============================================================================
    # Test model
    # =============================================================================
    model_test(model_saved_path, data_loaders)
    print(f'Model tested')

