"""
ISIC 2018 – Task 3: Skin Lesion Diagnosis
Dataset loader for PyTorch.

Estrutura esperada após o download (gerenciada automaticamente):
    <root>/
        train/
            images/          ← ISIC2018_Task3_Training_Input/
            ground_truth/    ← ISIC2018_Task3_Training_GroundTruth/
            metadata.csv     ← LesionGroupings
        val/
            images/
            ground_truth/
        test/
            images/
            ground_truth/

Classes (7):
    MEL, NV, BCC, AKIEC, BKL, DF, VASC

Uso básico:
    from isic2018_dataset import ISIC2018Dataset, download_isic2018, get_dataloaders

    download_isic2018(root="./data/isic2018")   # baixa e extrai tudo
    train_loader, val_loader, test_loader = get_dataloaders(root="./data/isic2018")
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

URLS: Dict[str, str] = {
    "train_images":    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_Input.zip",
    "train_gt":        "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip",
    "train_meta":      "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_LesionGroupings.csv",
    "val_images":      "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Validation_Input.zip",
    "val_gt":          "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Validation_GroundTruth.zip",
    "test_images":     "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_Input.zip",
    "test_gt":         "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_GroundTruth.zip",
}

# Ordem oficial das colunas no CSV de ground truth
CLASSES: List[str] = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Download e extração
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Baixa *url* para *dest* com barra de progresso. Retorna o caminho."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [cache] {dest.name} já existe, pulando download.")
        return dest

    print(f"  Baixando {url} …")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extrai *zip_path* em *out_dir* (cria se necessário)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extraindo {zip_path.name} → {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _find_image_dir(base: Path) -> Path:
    """
    Após extração o zip pode criar um sub-diretório com o mesmo nome do arquivo.
    Procura recursivamente pelo diretório que contém arquivos .jpg.
    """
    for candidate in sorted(base.rglob("*.jpg")):
        return candidate.parent
    raise FileNotFoundError(f"Nenhuma imagem .jpg encontrada em {base}")


def _find_gt_csv(base: Path) -> Path:
    """Localiza o CSV de ground truth dentro do diretório extraído."""
    candidates = list(base.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"CSV de ground truth não encontrado em {base}")
    # Prefere o que contém 'GroundTruth' no nome
    for c in candidates:
        if "GroundTruth" in c.name:
            return c
    return candidates[0]


def download_isic2018(root: str | Path = "./data/isic2018", keep_zips: bool = False) -> None:
    """
    Baixa e organiza o dataset ISIC 2018 Task 3 em *root*.

    Parâmetros
    ----------
    root : caminho base onde os dados serão salvos.
    keep_zips : se True, mantém os arquivos .zip após extração.
    """
    root = Path(root)
    cache = root / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": ("train_images", "train_gt", "train_meta"),
        "val":   ("val_images",   "val_gt",   None),
        "test":  ("test_images",  "test_gt",  None),
    }

    for split, (img_key, gt_key, meta_key) in splits.items():
        print(f"\n{'='*50}")
        print(f"  Split: {split.upper()}")
        print(f"{'='*50}")

        split_dir = root / split

        # --- Imagens ---
        img_zip = _download_file(URLS[img_key], cache / f"{img_key}.zip")
        img_raw = cache / f"{img_key}_extracted"
        if not img_raw.exists():
            _extract_zip(img_zip, img_raw)
        img_src = _find_image_dir(img_raw)
        img_dst = split_dir / "images"
        if not img_dst.exists():
            print(f"  Movendo imagens → {img_dst}")
            shutil.copytree(img_src, img_dst)

        # --- Ground truth ---
        gt_zip = _download_file(URLS[gt_key], cache / f"{gt_key}.zip")
        gt_raw = cache / f"{gt_key}_extracted"
        if not gt_raw.exists():
            _extract_zip(gt_zip, gt_raw)
        gt_src = _find_gt_csv(gt_raw)
        gt_dst = split_dir / "ground_truth.csv"
        if not gt_dst.exists():
            print(f"  Copiando ground truth → {gt_dst}")
            shutil.copy2(gt_src, gt_dst)

        # --- Metadados (apenas treino) ---
        if meta_key:
            meta_dst = split_dir / "metadata.csv"
            if not meta_dst.exists():
                meta_path = _download_file(URLS[meta_key], cache / "metadata.csv")
                shutil.copy2(meta_path, meta_dst)

    if not keep_zips:
        shutil.rmtree(cache, ignore_errors=True)

    print(f"\n✓ Download concluído. Dados em: {root.resolve()}")


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class ISIC2018Dataset(Dataset):
    """
    Dataset PyTorch para ISIC 2018 – Task 3.

    Parâmetros
    ----------
    root : caminho raiz onde o dataset foi organizado por `download_isic2018`.
    split : 'train' | 'val' | 'test'
    transform : transformações torchvision (aplicadas à imagem PIL).
    target_transform : transformação opcional sobre o label (int).
    return_metadata : se True e split=='train', o __getitem__ retorna
                      (image, label, meta_dict); caso contrário (image, label).

    Exemplo
    -------
    >>> ds = ISIC2018Dataset(root="./data/isic2018", split="train",
    ...                       transform=transforms.ToTensor())
    >>> img, label = ds[0]
    >>> print(ISIC2018Dataset.CLASSES[label])
    """

    CLASSES = CLASSES
    CLASS_TO_IDX = CLASS_TO_IDX

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ) -> None:
        assert split in ("train", "val", "test"), \
            f"split deve ser 'train', 'val' ou 'test'; recebido: '{split}'"

        self.root = Path(root) / split
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata

        # Lê ground truth
        gt_path = self.root / "ground_truth.csv"
        if not gt_path.exists():
            raise FileNotFoundError(
                f"{gt_path} não encontrado. Execute download_isic2018(root=...) primeiro."
            )
        self._gt = pd.read_csv(gt_path)

        # Coluna de identificador da imagem (normalmente 'image')
        id_col = self._gt.columns[0]
        self._image_ids: List[str] = self._gt[id_col].tolist()

        # Rótulo: índice da classe com valor 1
        label_cols = [c for c in self._gt.columns if c in CLASS_TO_IDX]
        self._labels: List[int] = (
            self._gt[label_cols].values.argmax(axis=1).tolist()
        )

        # Metadados opcionais (treino)
        self._meta: Optional[pd.DataFrame] = None
        meta_path = self.root / "metadata.csv"
        if meta_path.exists():
            self._meta = pd.read_csv(meta_path).set_index(
                pd.read_csv(meta_path).columns[0]
            )

        # Diretório de imagens
        self._img_dir = self.root / "images"

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int):
        img_id = self._image_ids[idx]
        label  = self._labels[idx]

        # Carrega imagem
        img_path = self._img_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_metadata and self._meta is not None:
            try:
                meta = self._meta.loc[img_id].to_dict()
            except KeyError:
                meta = {}
            return image, label, meta

        return image, label

    # ------------------------------------------------------------------
    def class_name(self, idx: int) -> str:
        """Retorna o nome da classe dado o índice."""
        return self.CLASSES[idx]

    def class_weights(self):
        """
        Retorna pesos inversamente proporcionais à frequência de cada classe.
        Útil para `WeightedRandomSampler` ou loss ponderada.

        Retorno: torch.Tensor de shape (num_classes,)
        """
        import torch

        counts = [0] * len(self.CLASSES)
        for lbl in self._labels:
            counts[lbl] += 1
        counts = torch.tensor(counts, dtype=torch.float)
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum()

    def __repr__(self) -> str:
        return (
            f"ISIC2018Dataset(split={self.split!r}, "
            f"n_samples={len(self)}, "
            f"classes={self.CLASSES})"
        )


# ---------------------------------------------------------------------------
# Transforms padrão
# ---------------------------------------------------------------------------

def default_transforms(
    split: str = "train",
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Retorna transforms padrão (ImageNet stats) para cada split.

    Treino inclui augmentação; val/test apenas resize + normalização.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ---------------------------------------------------------------------------
# Função de conveniência: get_dataloaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    root: str | Path = "./data/isic2018",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    weighted_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria e retorna (train_loader, val_loader, test_loader).

    Parâmetros
    ----------
    root : caminho raiz do dataset.
    image_size : dimensão espacial das imagens (quadrado).
    batch_size : tamanho do mini-batch.
    num_workers : workers para carregamento paralelo.
    pin_memory : habilita pin_memory (recomendado com GPU).
    train_transform / val_transform / test_transform : substituem os padrões.
    weighted_sampling : usa WeightedRandomSampler no treino para balancear classes.

    Retorno
    -------
    Tupla (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import WeightedRandomSampler
    import torch

    tf_train = train_transform or default_transforms("train", image_size)
    tf_val   = val_transform   or default_transforms("val",   image_size)
    tf_test  = test_transform  or default_transforms("test",  image_size)

    train_ds = ISIC2018Dataset(root, split="train", transform=tf_train)
    val_ds   = ISIC2018Dataset(root, split="val",   transform=tf_val)
    test_ds  = ISIC2018Dataset(root, split="test",  transform=tf_test)

    # Sampler ponderado para balancear classes no treino
    train_sampler = None
    if weighted_sampling:
        class_w = train_ds.class_weights()          # (num_classes,)
        sample_w = torch.tensor(
            [class_w[lbl].item() for lbl in train_ds._labels]
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(train_ds),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CLI mínimo: python isic2018_dataset.py --root ./data/isic2018
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baixa o dataset ISIC 2018 Task 3.")
    parser.add_argument("--root", default="./data/isic2018",
                        help="Diretório onde os dados serão salvos.")
    parser.add_argument("--keep-zips", action="store_true",
                        help="Mantém arquivos .zip após extração.")
    args = parser.parse_args()

    download_isic2018(root=args.root, keep_zips=args.keep_zips)

    # Smoke-test rápido
    print("\n--- Smoke test ---")
    for split in ("train", "val", "test"):
        ds = ISIC2018Dataset(root=args.root, split=split)
        print(ds)
        img, lbl = ds[0]
        print(f"  Primeira amostra: PIL Image {img.size}, label={lbl} ({ds.class_name(lbl)})")
