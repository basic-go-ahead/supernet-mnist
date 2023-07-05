import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.models import admissible_architectures, SuperNet
from src.utils import fix_seed
from src.utils.samplers import UniformSampler, EpsilonGreedySampler


def train_epoch(
    current_epoch: int,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    verbose: bool = True
):
    model.train()
    error = nn.CrossEntropyLoss()
    total_loss = 0.
    handled = 0

    with tqdm(loader) as progress_bar:
        for batch_index, (images, labels) in enumerate(progress_bar, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            output = model(images)

            if model.sampler is not None:
                prediction = torch.max(output, 1)[1]
                right = (prediction == labels).sum().item()
                model.sampler.feedback(right, len(labels))

            loss = error(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            handled += len(labels)
            if verbose and batch_index % 10 == 0:
                progress_bar.set_description(f"Epoch {current_epoch:02d}: mean_train_loss = {total_loss / batch_index:.05f}")

    return total_loss / batch_index


def main():
    #region Command Line Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=str, required=True, help="путь к CSV-файлу обучающего набора данных")
    parser.add_argument("--epochs", type=int, required=True, help="кол-во эпох обучения")
    parser.add_argument("--batch-size", type=int, default=32, help="размер батча (по умолчанию 32)")
    parser.add_argument("--model-path", type=str, required=True, help="путь к файлу, в который следует сохранить обученную модель")
    parser.add_argument("--sampler-history-path", type=str,
        help="путь к директории, в которую следует сохранить историю фидбека, накопленную сэмплером"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sampler", type=str,
        help="тип сэмплера для обучения суперсети (пока только `uniform`)",
        choices=["uniform", "epsilon-greedy"]
    )
    group.add_argument("--architecture", nargs="+", type=int,
        help="тип конкретной архитектуры подсети суперсети (два значения через пробел)"
    )

    args = vars(parser.parse_args())
    #endregion

    EPOCHS = args["epochs"]
    DATASET_PATH = args["dataset_path"]
    BATCH_SIZE = args["batch_size"]
    MODEL_PATH = args["model_path"]
    SAMPLER = args["sampler"]
    ARCHITECTURE = args["architecture"]
    SAMPLER_HISTORY_PATH = args["sampler_history_path"]

    fix_seed(1187)
    
    if SAMPLER:
        model = SuperNet(
            sampler=UniformSampler(admissible_architectures) if SAMPLER == "uniform" else EpsilonGreedySampler(
                admissible_architectures,
                0.2
            )
        )
        print(f"\033[34mTraining SuperNet with {SAMPLER} sampler...\033[0m")
    else:
        model = SuperNet(architecture=tuple(ARCHITECTURE))
        print(f"\033[34mTraining Arch({ARCHITECTURE[0]}, {ARCHITECTURE[1]})...\033[0m")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataset = MNIST(DATASET_PATH, train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for current_epoch in range(1, EPOCHS + 1):
        train_epoch(current_epoch, model, train_loader, optimizer, device)
        
    torch.save(model, MODEL_PATH)

    if SAMPLER and SAMPLER_HISTORY_PATH:
        Path(SAMPLER_HISTORY_PATH).mkdir(exist_ok=True)
        model.sampler.save(SAMPLER_HISTORY_PATH)


if __name__ == "__main__":
    main()