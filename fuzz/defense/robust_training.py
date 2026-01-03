import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from art.estimators.classification import PyTorchClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD, AdversarialTrainerFBFPyTorch
from art.attacks.evasion import ProjectedGradientDescent
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--dataset_target', type=str, default='DSAN/datasets/office31', help='Path to the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='Nb of epochs for training')
parser.add_argument('--model_target', type=str, default='densenet', help='The target model to test')
parser.add_argument('--adv_trainer', type=str, default='madry', help='The adversarial training method to use')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use (or "cpu" to use CPU)')
args = parser.parse_args()

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

if "Product" in args.dataset_target or "RealWorld" in args.dataset_target:
    num_classes = 65
    data_path = f"./datasets/office_home/{args.dataset_target}"
    model_path = f"DSAN/models/target_{str(args.dataset_target)}_{args.model_target}.pt"
    root_path_save = f"defense/models/office_home"
else:
    num_classes = 31
    data_path = f"./datasets/office31/{args.dataset_target}"
    model_path = f"DSAN/models/target_{str(args.dataset_target)}_{args.model_target}.pt"
    root_path_save = f"defense/models/office31"


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



def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


# Initialize the model
model = torch.load(model_path, weights_only=False, map_location=device)
model = model.to(device)

model.train()

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

input_size = 224  # Change as needed
data_loaders, dataset_sizes = load_data(input_size)


classifier = PyTorchClassifier(
    model=model,
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(3, input_size, input_size),
    nb_classes=num_classes,
    clip_values=(0, 1),
    device_type='gpu' if torch.cuda.is_available() else 'cpu'
)

def data_loader_to_numpy(data_loader):
    data = []
    labels = []
    for inputs, labels_batch in data_loader:
        data.append(inputs.numpy())
        labels.append(labels_batch.numpy())
    return np.concatenate(data), np.concatenate(labels)

x_train, y_train = data_loader_to_numpy(data_loaders['train'])
x_test, y_test = data_loader_to_numpy(data_loaders['test'])



y_train_one_hot = to_one_hot(y_train, num_classes)
y_test_one_hot = to_one_hot(y_test, num_classes)


if str(args.adv_trainer).lower() == 'madry':
    print("Starting adversarial training using Madry's protocol...")
    trainer_madry = AdversarialTrainerMadryPGD(classifier, nb_epochs=args.epochs, batch_size=args.batch_size, eps=8/255, eps_step=2/255, max_iter=7)

    trainer_madry.fit(x_train, y_train)

    # Evaluate on clean test examples
    predictions = classifier.predict(x_test)
    accuracy_madry = np.sum(np.argmax(predictions, axis=1) == y_test) / y_test.shape[0]
    print("Test accuracy after Madry's adversarial training: {:.2f}%".format(accuracy_madry * 100))

    # save the model
    if not os.path.exists(root_path_save):
        os.makedirs(root_path_save)
    torch.save(classifier.model, f"{root_path_save}/target_{str(args.dataset_target).lower()}_{args.model_target}_adv_{args.adv_trainer}.pt")

elif str(args.adv_trainer).lower() == 'fbf':
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(3, input_size, input_size),
        nb_classes=num_classes,
        clip_values=(0, 1),
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )

    trainer_fbf = AdversarialTrainerFBFPyTorch(classifier, eps=8/255)
    trainer_fbf.fit(x_train, y_train_one_hot, nb_epochs=args.epochs, batch_size=args.batch_size)

    predictions = classifier.predict(x_test)
    accuracy_fbf = np.sum(np.argmax(predictions, axis=1) == y_test) / y_test.shape[0]
    print("Test accuracy after FBF adversarial training: {:.2f}%".format(accuracy_fbf * 100))
    if not os.path.exists(root_path_save):
        os.makedirs(root_path_save)
    torch.save(classifier.model, f"{root_path_save}/target_{str(args.dataset_target).lower()}_{args.model_target}_adv_{args.adv_trainer}.pt")

