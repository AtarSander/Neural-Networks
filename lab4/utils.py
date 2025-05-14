import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import utils
from collections import Counter
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import TensorDataset, DataLoader


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def to_displayable(img):
    img = img / 2 + 0.5
    return img.clamp(0, 1)


def visualize_images(dataloader, classes, figsize=(20, 10), batch_size=16):
    plt.figure(figsize=figsize)
    images, labels = next(iter(dataloader))
    images, labels = next(iter(dataloader))

    imshow(utils.make_grid(images))

    print(" ".join("%5s" % classes[labels[j]] for j in range(batch_size)))


def verify_image_sizes(dataloader):
    sizes = {}
    for images, _ in dataloader:
        for image in images:
            size = image.size()
            sizes[size] = sizes.get(size, 0) + 1
    return sizes


def plot_bar(values, title, size=(10, 4)):
    plt.figure(figsize=size)
    plt.bar(*zip(*values.items()))
    plt.title(title)
    plt.show()


def create_conditioning_distribution(dataset, sample_size=1000):
    frequency = Counter(dataset.targets)
    total_count = len(dataset)
    exact = {
        label: (count / total_count) * sample_size for label, count in frequency.items()
    }
    floored = {label: int(np.floor(count)) for label, count in exact.items()}
    remainder = sample_size - sum(floored.values())
    fractional_parts = {label: count - floored[label] for label, count in exact.items()}
    sorted_labels = sorted(fractional_parts, key=fractional_parts.get, reverse=True)
    for i in range(remainder):
        floored[sorted_labels[i]] += 1
    return floored


def create_conditioning_vector(frequency):
    conditioning_vector = []
    for label, count in frequency.items():
        conditioning_vector.extend([label] * count)
    return torch.tensor(conditioning_vector)


def show_generated(generated_images, range_low=0, range_high=-1):
    imshow(utils.make_grid(generated_images[range_low:range_high]))


def to_uint8(img):
    img = (img + 1) / 2
    img = img.clamp(0, 1)
    img = (img * 255).round()
    return img.to(torch.uint8)


def measure_fid(real_dataset, tensor_fake, device="cuda"):
    real_loader = DataLoader(
        real_dataset, batch_size=64, shuffle=False, num_workers=7, pin_memory=True
    )
    gen_dataset = TensorDataset(tensor_fake.cpu())
    gen_loader = DataLoader(
        gen_dataset, batch_size=64, shuffle=False, num_workers=7, pin_memory=True
    )

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.eval()
    with torch.no_grad():
        for (imgs,) in gen_loader:
            fid.update(to_uint8(imgs).to(device), real=True)

        for imgs, label in real_loader:
            fid.update(to_uint8(imgs).to(device), real=False)

    score = fid.compute()
    fid.reset()
    return score.item()


def visualize_images_grid(images, figsize=(20, 10), batch_size=16, title=None):
    fig, axes = plt.subplots(len(images), 1, figsize=figsize)
    for i, (name, image) in enumerate(images.items()):
        if isinstance(image, torch.Tensor):
            image = TensorDataset(image)
        loader = DataLoader(image, batch_size=batch_size, shuffle=True)
        image = next(iter(loader))[0]
        grid = to_displayable(utils.make_grid(image))
        grid = grid.permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(grid)
        axes[i].set_title(name)
        axes[i].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
