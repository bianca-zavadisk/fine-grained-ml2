"""
ISIC 2018 – Task 3: Skin Lesion Diagnosis  (HAM10000)
Dataset loader for PyTorch.

Estrutura esperada após o download (gerenciada automaticamente):
    <root>/
        train/
            images/          ← ISIC2018_Task3_Training_Input/
            ground_truth.csv ← ISIC2018_Task3_Training_GroundTruth/
            metadata.csv     ← LesionGroupings
        val/
            images/
            ground_truth.csv
        test/
            images/
            ground_truth.csv

Classes (7):
    MEL, NV, BCC, AKIEC, BKL, DF, VASC

Uso básico:
    from isic2018_dataset import (
        download_isic2018,
        get_dataloaders,
        analyze_lesion_groups,
        lesion_group_summary,
        get_dataloaders_leak_free,
    )

    download_isic2018(root="./data/isic2018")
    summary = lesion_group_summary(root="./data/isic2018")

    # Split sem data leakage (recomendado)
    train_loader, val_loader, test_loader = get_dataloaders_leak_free(
        root="./data/isic2018"
    )
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

URLS: Dict[str, str] = {
    "train_images": "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_Input.zip",
    "train_gt":     "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip",
    "train_meta":   "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_LesionGroupings.csv",
    "val_images":   "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Validation_Input.zip",
    "val_gt":       "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Validation_GroundTruth.zip",
    "test_images":  "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_Input.zip",
    "test_gt":      "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_GroundTruth.zip",
}

CLASSES: List[str] = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Helpers de download / extração
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
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
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extraindo {zip_path.name} → {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _find_image_dir(base: Path) -> Path:
    for candidate in sorted(base.rglob("*.jpg")):
        return candidate.parent
    raise FileNotFoundError(f"Nenhuma imagem .jpg encontrada em {base}")


def _find_gt_csv(base: Path) -> Path:
    candidates = list(base.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"CSV de ground truth não encontrado em {base}")
    for c in candidates:
        if "GroundTruth" in c.name:
            return c
    return candidates[0]


def _is_dataset_prepared(root: Path) -> bool:
    """
    Retorna True se o dataset já estiver organizado em root com os três splits
    (cada split contendo 'images/' e 'ground_truth.csv').
    """
    for s in ("train", "val", "test"):
        split_dir = root / s
        if not (split_dir / "images").exists() or not (split_dir / "ground_truth.csv").exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Download principal
# ---------------------------------------------------------------------------

def download_isic2018(root: str | Path = "./data/isic2018", keep_zips: bool = False) -> None:
    """
    Baixa e organiza o dataset ISIC 2018 Task 3 em *root*.
    Operação idempotente: se os dados já existirem, retorna imediatamente.
    """
    root = Path(root)

    if _is_dataset_prepared(root):
        print(f"✓ Dados já preparados em: {root.resolve()} — pulando download/extração.")
        return

    cache = root / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": ("train_images", "train_gt", "train_meta"),
        "val":   ("val_images",   "val_gt",   None),
        "test":  ("test_images",  "test_gt",  None),
    }

    for split, (img_key, gt_key, meta_key) in splits.items():
        print(f"\n{'='*50}\n  Split: {split.upper()}\n{'='*50}")
        split_dir = root / split

        img_zip = _download_file(URLS[img_key], cache / f"{img_key}.zip")
        img_raw = cache / f"{img_key}_extracted"
        if not img_raw.exists():
            _extract_zip(img_zip, img_raw)
        img_src = _find_image_dir(img_raw)
        img_dst = split_dir / "images"
        if not img_dst.exists():
            print(f"  Movendo imagens → {img_dst}")
            shutil.copytree(img_src, img_dst)

        gt_zip = _download_file(URLS[gt_key], cache / f"{gt_key}.zip")
        gt_raw = cache / f"{gt_key}_extracted"
        if not gt_raw.exists():
            _extract_zip(gt_zip, gt_raw)
        gt_src = _find_gt_csv(gt_raw)
        gt_dst = split_dir / "ground_truth.csv"
        if not gt_dst.exists():
            print(f"  Copiando ground truth → {gt_dst}")
            shutil.copy2(gt_src, gt_dst)

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
    root : caminho raiz organizado por `download_isic2018`.
    split : 'train' | 'val' | 'test'
    transform : transformações torchvision (aplicadas à imagem PIL).
    target_transform : transformação opcional sobre o label (int).
    return_metadata : se True e split=='train', __getitem__ retorna
                      (image, label, meta_dict).
    """

    CLASSES     = CLASSES
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

        self.root             = Path(root) / split
        self.split            = split
        self.transform        = transform
        self.target_transform = target_transform
        self.return_metadata  = return_metadata

        gt_path = self.root / "ground_truth.csv"
        if not gt_path.exists():
            raise FileNotFoundError(
                f"{gt_path} não encontrado. Execute download_isic2018(root=...) primeiro."
            )
        self._gt = pd.read_csv(gt_path)

        id_col = self._gt.columns[0]
        self._image_ids: List[str] = self._gt[id_col].tolist()

        label_cols      = [c for c in self._gt.columns if c in CLASS_TO_IDX]
        self._labels: List[int] = self._gt[label_cols].values.argmax(axis=1).tolist()

        self._meta: Optional[pd.DataFrame] = None
        meta_path = self.root / "metadata.csv"
        if meta_path.exists():
            _m = pd.read_csv(meta_path)
            self._meta = _m.set_index(_m.columns[0])

        self._img_dir = self.root / "images"

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int):
        img_id = self._image_ids[idx]
        label  = self._labels[idx]

        image = Image.open(self._img_dir / f"{img_id}.jpg").convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_metadata and self._meta is not None:
            meta = self._meta.loc[img_id].to_dict() if img_id in self._meta.index else {}
            return image, label, meta

        return image, label

    def class_name(self, idx: int) -> str:
        return self.CLASSES[idx]

    def class_weights(self) -> torch.Tensor:
        """Pesos inversamente proporcionais à frequência — útil para WeightedRandomSampler."""
        counts = torch.zeros(len(self.CLASSES))
        for lbl in self._labels:
            counts[lbl] += 1
        w = 1.0 / counts.clamp(min=1)
        return w / w.sum()

    def __repr__(self) -> str:
        return (
            f"ISIC2018Dataset(split={self.split!r}, "
            f"n_samples={len(self)}, classes={self.CLASSES})"
        )


# ---------------------------------------------------------------------------
# Análise de grupos de lesão (LesionGroupings)
# ---------------------------------------------------------------------------

def analyze_lesion_groups(root: str | Path) -> pd.DataFrame:
    """
    Analisa o arquivo LesionGroupings e retorna um DataFrame no nível da
    **lesão** (não da imagem), com as colunas:

        lesion_id | diagnosis | n_images | is_unique

    Parâmetros
    ----------
    root : caminho raiz do dataset (deve conter train/metadata.csv).

    Retorno
    -------
    DataFrame indexado por lesion_id com informações de agrupamento.

    Notas
    -----
    O arquivo LesionGroupings mapeia cada image_id → lesion_id.
    Lesões com n_images > 1 são variações de ângulo/zoom da mesma lesão.
    Essas variações NÃO devem cruzar os splits treino/validação para evitar
    data leakage.
    """
    root = Path(root)
    meta_path = root / "train" / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} não encontrado. Execute download_isic2018(root=...) primeiro."
        )

    gt_path = root / "train" / "ground_truth.csv"
    meta = pd.read_csv(meta_path)
    gt   = pd.read_csv(gt_path)

    # Normaliza nomes de coluna
    img_col    = meta.columns[0]       # e.g. 'image'
    lesion_col = meta.columns[1]       # e.g. 'lesion_id'

    # Junta ground truth para obter o diagnóstico de cada imagem
    label_cols = [c for c in gt.columns if c in CLASS_TO_IDX]
    gt["diagnosis"] = gt[label_cols].idxmax(axis=1)
    merged = meta[[img_col, lesion_col]].merge(
        gt[[gt.columns[0], "diagnosis"]], left_on=img_col, right_on=gt.columns[0]
    )

    # Agrega por lesão
    lesion_df = (
        merged.groupby(lesion_col)
        .agg(
            diagnosis=("diagnosis", "first"),
            n_images=(img_col, "count"),
            image_ids=(img_col, list),
        )
        .reset_index()
    )
    lesion_df["is_unique"] = lesion_df["n_images"] == 1
    lesion_df = lesion_df.rename(columns={lesion_col: "lesion_id"})
    return lesion_df.set_index("lesion_id")


def lesion_group_summary(root: str | Path) -> pd.DataFrame:
    """
    Exibe e retorna uma tabela-resumo por classe com as seguintes colunas:

        diagnosis | total_images | unique_lesions | multi_image_lesions
                  | images_in_multi | max_views | mean_views

    Parâmetros
    ----------
    root : caminho raiz do dataset.

    Retorno
    -------
    DataFrame com uma linha por classe diagnóstica.
    """
    lg = analyze_lesion_groups(root).reset_index()

    summary_rows = []
    for cls in CLASSES:
        sub = lg[lg["diagnosis"] == cls]
        unique  = sub[sub["is_unique"]]
        multi   = sub[~sub["is_unique"]]

        summary_rows.append({
            "diagnosis":           cls,
            "total_images":        int(sub["n_images"].sum()),
            "unique_lesions":      len(unique),
            "multi_image_lesions": len(multi),
            "images_in_multi":     int(multi["n_images"].sum()),
            "max_views":           int(sub["n_images"].max()) if len(sub) else 0,
            "mean_views":          round(sub["n_images"].mean(), 2) if len(sub) else 0.0,
        })

    df = pd.DataFrame(summary_rows).set_index("diagnosis")

    # Linha totalizadora
    totals = {
        "total_images":        df["total_images"].sum(),
        "unique_lesions":      df["unique_lesions"].sum(),
        "multi_image_lesions": df["multi_image_lesions"].sum(),
        "images_in_multi":     df["images_in_multi"].sum(),
        "max_views":           df["max_views"].max(),
        "mean_views":          round(df["mean_views"].mean(), 2),
    }
    df.loc["TOTAL"] = totals

    print("\n" + "=" * 72)
    print("  Resumo de grupos de lesão – ISIC 2018 Task 3 (split treino)")
    print("=" * 72)
    print(df.to_string())
    print("=" * 72)
    print(
        "\nLegenda:\n"
        "  total_images        → imagens totais da classe\n"
        "  unique_lesions      → lesões com apenas 1 imagem\n"
        "  multi_image_lesions → lesões com ≥2 imagens (risco de leakage)\n"
        "  images_in_multi     → imagens pertencentes a grupos multi-view\n"
        "  max_views           → máximo de imagens por lesão\n"
        "  mean_views          → média de imagens por lesão\n"
    )
    return df


# ---------------------------------------------------------------------------
# Split estratificado sem data leakage
# ---------------------------------------------------------------------------

def _lesion_aware_split(
    lesion_df: pd.DataFrame,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Divide as **lesões** (não imagens) em conjuntos treino e validação de
    forma estratificada por diagnóstico, garantindo que todas as imagens de
    uma mesma lesão fiquem no mesmo split.

    Retorna (train_image_ids, val_image_ids).
    """
    from sklearn.model_selection import GroupShuffleSplit

    lg = lesion_df.reset_index()

    # GroupShuffleSplit: grupos = lesion_id, estratificação manual por classe
    train_ids: List[str] = []
    val_ids:   List[str] = []

    for cls in CLASSES:
        sub = lg[lg["diagnosis"] == cls].copy()
        if len(sub) == 0:
            continue

        n_val = max(1, int(round(len(sub) * val_size)))

        # Embaralha e separa lesões (não imagens)
        sub_shuffled = sub.sample(frac=1, random_state=seed)
        val_lesions  = sub_shuffled.iloc[:n_val]
        train_lesions = sub_shuffled.iloc[n_val:]

        for _, row in train_lesions.iterrows():
            train_ids.extend(row["image_ids"])
        for _, row in val_lesions.iterrows():
            val_ids.extend(row["image_ids"])

    return train_ids, val_ids


def get_dataloaders_leak_free(
    root: str | Path = "./data/isic2018",
    val_size: float = 0.15,
    seed: int = 42,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform:   Optional[Callable] = None,
    test_transform:  Optional[Callable] = None,
    weighted_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria dataloaders de treino, validação e teste garantindo que imagens da
    mesma lesão (multi-view) nunca cruzem os splits treino/validação.

    Estratégia
    ----------
    1. Carrega o LesionGroupings para identificar grupos de lesão.
    2. Divide **lesões** (não imagens) de forma estratificada por classe.
    3. Constrói Subsets do ISIC2018Dataset com os image_ids resultantes.
    4. O split de teste usa os dados oficiais de teste do ISIC 2018 (inalterados).

    Parâmetros
    ----------
    root            : caminho raiz do dataset.
    val_size        : fração das lesões destinada à validação (default 0.15).
    seed            : semente para reprodutibilidade.
    image_size      : dimensão espacial das imagens.
    batch_size      : tamanho do mini-batch.
    num_workers     : workers para carregamento paralelo.
    pin_memory      : habilita pin_memory (recomendado com GPU).
    train_transform / val_transform / test_transform : substituem os padrões.
    weighted_sampling : WeightedRandomSampler no treino para balancear classes.

    Retorno
    -------
    (train_loader, val_loader, test_loader)
    """
    root = Path(root)

    tf_train = train_transform or default_transforms("train", image_size)
    tf_val   = val_transform   or default_transforms("val",   image_size)
    tf_test  = test_transform  or default_transforms("test",  image_size)

    # --- Carrega o dataset de treino completo (sem transform por enquanto) ---
    base_ds = ISIC2018Dataset(root, split="train")

    # --- Análise de grupos ---
    lesion_df = analyze_lesion_groups(root)
    train_img_ids, val_img_ids = _lesion_aware_split(lesion_df, val_size=val_size, seed=seed)

    # Mapeia image_id → índice no dataset
    id_to_idx = {img_id: i for i, img_id in enumerate(base_ds._image_ids)}

    train_indices = [id_to_idx[i] for i in train_img_ids if i in id_to_idx]
    val_indices   = [id_to_idx[i] for i in val_img_ids   if i in id_to_idx]

    # --- Datasets com transforms corretos ---
    train_ds_tf = ISIC2018Dataset(root, split="train", transform=tf_train)
    val_ds_tf   = ISIC2018Dataset(root, split="train", transform=tf_val)

    train_subset = Subset(train_ds_tf, train_indices)
    val_subset   = Subset(val_ds_tf,   val_indices)

    # --- Sampler ponderado (baseado apenas nas amostras de treino) ---
    train_sampler = None
    if weighted_sampling:
        train_labels = [base_ds._labels[i] for i in train_indices]
        counts = torch.zeros(len(CLASSES))
        for lbl in train_labels:
            counts[lbl] += 1
        class_w  = 1.0 / counts.clamp(min=1)
        sample_w = torch.tensor([class_w[lbl].item() for lbl in train_labels])
        train_sampler = WeightedRandomSampler(
            weights=sample_w, num_samples=len(train_indices), replacement=True
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # --- Teste: usa o split oficial inalterado ---
    test_ds = ISIC2018Dataset(root, split="test", transform=tf_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Relatório
    train_labels_final = [base_ds._labels[i] for i in train_indices]
    val_labels_final   = [base_ds._labels[i] for i in val_indices]
    _report_split(train_labels_final, val_labels_final, list(test_ds._labels))

    return train_loader, val_loader, test_loader


def _report_split(
    train_labels: List[int],
    val_labels:   List[int],
    test_labels:  List[int],
) -> None:
    """Imprime distribuição de classes nos três splits."""
    from collections import Counter

    print("\n" + "=" * 60)
    print("  Distribuição de classes após split sem data leakage")
    print("=" * 60)
    header = f"{'Classe':<10} {'Treino':>8} {'Val':>8} {'Teste':>8}"
    print(header)
    print("-" * 40)

    c_tr = Counter(train_labels)
    c_va = Counter(val_labels)
    c_te = Counter(test_labels)
    for idx, cls in enumerate(CLASSES):
        print(f"{cls:<10} {c_tr[idx]:>8} {c_va[idx]:>8} {c_te[idx]:>8}")
    print("-" * 40)
    print(f"{'TOTAL':<10} {len(train_labels):>8} {len(val_labels):>8} {len(test_labels):>8}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Transforms padrão
# ---------------------------------------------------------------------------

def default_transforms(
    split: str = "train",
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std:  Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Transforms padrão com ImageNet stats.
    Treino: augmentação + normalização.
    Val/Teste: apenas resize + normalização.
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
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# get_dataloaders (split original, sem agrupamento de lesões)
# ---------------------------------------------------------------------------

def get_dataloaders(
    root: str | Path = "./data/isic2018",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Callable] = None,
    val_transform:   Optional[Callable] = None,
    test_transform:  Optional[Callable] = None,
    weighted_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Versão simples: usa os splits originais do ISIC 2018 sem rebalancear por
    grupo de lesão. Prefira `get_dataloaders_leak_free` para evitar leakage.
    """
    tf_train = train_transform or default_transforms("train", image_size)
    tf_val   = val_transform   or default_transforms("val",   image_size)
    tf_test  = test_transform  or default_transforms("test",  image_size)

    train_ds = ISIC2018Dataset(root, split="train", transform=tf_train)
    val_ds   = ISIC2018Dataset(root, split="val",   transform=tf_val)
    test_ds  = ISIC2018Dataset(root, split="test",  transform=tf_test)

    train_sampler = None
    if weighted_sampling:
        class_w  = train_ds.class_weights()
        sample_w = torch.tensor([class_w[lbl].item() for lbl in train_ds._labels])
        train_sampler = WeightedRandomSampler(
            weights=sample_w, num_samples=len(train_ds), replacement=True
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ISIC 2018 Task 3 – download e análise.")
    parser.add_argument("--root", default="./data/isic2018")
    parser.add_argument("--keep-zips", action="store_true")
    parser.add_argument("--summary", action="store_true",
                        help="Exibe tabela de grupos de lesão após o download.")
    args = parser.parse_args()

    download_isic2018(root=args.root, keep_zips=args.keep_zips)

    if args.summary:
        lesion_group_summary(root=args.root)

    print("\n--- Smoke test (split sem leakage) ---")
    tr, va, te = get_dataloaders_leak_free(root=args.root, batch_size=8, num_workers=0)
    imgs, lbls = next(iter(tr))
    print(f"Batch treino  : {imgs.shape}, labels={lbls.tolist()}")