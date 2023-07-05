import argparse
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.models import admissible_architectures, ForwardOptions
from src.utils import fix_seed


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str, **forwardOptions: ForwardOptions):
    model.eval()

    t, s = 0, 0

    with torch.no_grad(), tqdm(loader) as progress_bar:
        notify = lambda: progress_bar.set_description(f"Top-1 Acc = {s / t:.05f}")
        for batch_index, (images, labels) in enumerate(progress_bar, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(images, **forwardOptions)
            prediction = torch.max(output, 1)[1]
            
            s += (prediction == labels).sum().item()
            t += len(labels)

            if batch_index % 100 == 0:
                notify()
        
        notify()

    return s / t


def main():
    #region Command Line Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=str, required=True, help="путь к CSV-файлу набора данных")
    parser.add_argument("--batch-size", type=int, default=32, help="размер батча (по умолчанию 32)")
    parser.add_argument("--model-path", type=str, required=True, help="путь к файлу модели")

    args = vars(parser.parse_args())
    #endregion

    MODEL_PATH = args["model_path"]
    DATASET_PATH = args["dataset_path"]
    BATCH_SIZE = args["batch_size"]

    fix_seed(1187)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(MODEL_PATH, map_location=device)

    dataset = MNIST(DATASET_PATH, train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

    logging.basicConfig(filename="eval.log", encoding="utf-8", level=logging.INFO, force=True)

    if model.sampler is None:
        print(f"\033[34mEvaluating Arch({model.architecture[0]}, {model.architecture[1]})...\033[0m")
        v = evaluate(model, loader, device)
        logging.info(f"subnet\t{model.architecture}\t{v}")
    else:
        for architecture in admissible_architectures:
            print(f"\033[34mEvaluating Arch({architecture[0]}, {architecture[1]})...\033[0m")
            v = evaluate(model, loader, device, architecture=architecture)
            logging.info(f"{model.sampler.name}\t{architecture}\t{v}")


if __name__ == "__main__":
    main()