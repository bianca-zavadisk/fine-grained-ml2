import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, balanced_accuracy_score, auc
from sklearn.preprocessing import label_binarize
from modules.pipeline import evaluate_classification
from modules.binary_pipeline import GenericEarlyStopping, train_one_epoch
from modules.visualizations import roc_curve, log_pr_curve_multiclass, plot_top_errors_multiclass, log_gradcam_tensorboard

class_names = [
    "MEL",
    "NV",
    "BCC",
    "AKIEC",
    "BKL",
    "DF",
    "VASC"
]

@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0

    all_targets = []
    all_preds = []
    all_probs = []

    for inputs, labels in loader:

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)

        probs = torch.softmax(outputs, dim=1)

        preds = probs.argmax(dim=1)

        all_targets.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    num_classes = y_prob.shape[1]

    from sklearn.preprocessing import label_binarize
    # One-hot labels for ROC-AUC and PR-AUC
    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(num_classes)
    )

    # -------------------------
    # ROC-AUC
    # -------------------------
    try:
        roc_auc = roc_auc_score(
            y_true_bin,
            y_prob,
            average="macro",
            multi_class="ovr"
        )
    except ValueError:
        roc_auc = 0.0

    # -------------------------
    # PR-AUC
    # -------------------------
    try:
        pr_auc = average_precision_score(
            y_true_bin,
            y_prob,
            average="macro"
        )
    except ValueError:
        pr_auc = 0.0

    # -------------------------
    # Per-class metrics
    # -------------------------
    precision_per_class = precision_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    recall_per_class = recall_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    f1_per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    metrics = {

        # losses
        "loss": avg_loss,

        # accuracies
        "acc_global": (y_true == y_pred).mean(),
        "acc_ponderada": balanced_accuracy_score(
            y_true,
            y_pred
        ),

        # macro metrics
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        # aucs
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        # per-class arrays
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,

        # raw outputs
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    return metrics

def log_roc_curve_multiclass(
    y_true,
    y_prob,
    class_names=None
):
    """
    Plots one-vs-rest ROC curves for a multiclass classifier.

    Parameters
    ----------
    y_true : ndarray (N,)
        True labels

    y_prob : ndarray (N, C)
        Predicted probabilities

    class_names : list[str]
        Names of the classes

    Returns
    -------
    matplotlib.figure.Figure
    """

    num_classes = y_prob.shape[1]

    if class_names is None:
        class_names = [
            f"Class {i}"
            for i in range(num_classes)
        ]

    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(num_classes)
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    roc_aucs = []

    for i in range(num_classes):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i], # type: ignore
            y_prob[:, i]
        )

        roc_auc = auc(fpr, tpr)

        roc_aucs.append(roc_auc)

        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{class_names[i]} (AUC={roc_auc:.3f})"
        )

    macro_auc = np.mean(roc_aucs)

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        alpha=0.7,
        color="black",
        label="Random"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        f"Multiclass ROC Curves\nMacro AUC = {macro_auc:.3f}"
    )

    ax.legend(
        loc="lower right",
        fontsize=8
    )

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig

def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, model_type="", use_early_stopping=True, patience=5, log_every=5,student_run_tag='', output_dir='finalProject_outputs', mean=None, std=None, class_names=class_names, initial_epoch=0, writer=None):
    from torch.utils.tensorboard import SummaryWriter 
    if writer is None:
        writer = SummaryWriter(log_dir=f'./{output_dir}/{student_run_tag}/runs/{model_type}')

    history = {
        'train_loss': [], 'train_acc_global': [], 'train_acc_ponderada': [],
        'val_loss': [], 'val_acc_global': [], 'val_acc_ponderada': []
    }

    if use_early_stopping:
        early_stopping = GenericEarlyStopping(patience=patience, mode='max', path=f'./{output_dir}/{student_run_tag}/best_model_{model_type}.pth')

    def _confusion_matrix_figure(y_true, y_pred, class_map=None, normalize=True, cmap='Blues'):
        import numpy as _np
        import matplotlib.pyplot as _plt
        from sklearn.metrics import confusion_matrix as _confusion_matrix

        classes = _np.unique(_np.concatenate([y_true, y_pred]))
        if len(classes) == 0:
            fig = _plt.figure(figsize=(4, 3))
            _plt.text(0.5, 0.5, "No data", ha='center', va='center')
            return fig

        cm = _confusion_matrix(y_true, y_pred, labels=classes, normalize='true' if normalize else None)

        fig, ax = _plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if class_map is not None:
            tick_labels = [class_map.get(int(c), str(int(c))) for c in classes]
        else:
            tick_labels = [str(int(c)) for c in classes]

        ax.set_xticks(_np.arange(len(classes)))
        ax.set_yticks(_np.arange(len(classes)))
        ax.set_xticklabels(tick_labels, rotation=45)
        ax.set_yticklabels(tick_labels)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                s = f"{val:{fmt}}" if normalize else f"{int(val)}"
                ax.text(j, i, s, ha="center", va="center",
                        color="white" if val > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        fig.tight_layout()
        return fig

    for epoch in range(initial_epoch, initial_epoch + epochs):
        local_epoch = epoch - initial_epoch
        # 1. Executa Treinamento
        train_loss, train_acc_global, train_acc_pond = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch_idx=epoch+1, writer=writer, model_type=model_type
        )
        
        # 2. Executa Validação
        val_metrics = evaluate(model, val_loader, criterion, device)

        # 3. Atualiza Histórico
        history['train_loss'].append(train_loss)
        history['train_acc_global'].append(train_acc_global)
        history['train_acc_ponderada'].append(train_acc_pond)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc_global'].append(val_metrics['acc_global'])
        history['val_acc_ponderada'].append(val_metrics['acc_ponderada'])

        if ((epoch + 1) % log_every == 0) or (epoch + 1 == epochs) or (epoch + 1 == 1):
            print(f"Epoch {epoch+1}/{initial_epoch + epochs}:: "
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc_global:.4f} | Acc Pond.: {train_acc_pond:.4f} "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc Global: {val_metrics['acc_global']:.4f} | "
                f"Val Acc Pond: {val_metrics['acc_ponderada']:.4f}")

        if use_early_stopping:
            early_stopping(val_metrics['pr_auc'], model) # pyright: ignore[reportPossiblyUnboundVariable]

        # 4. Gravação no TensorBoard
        writer.add_scalars(f'{model_type}/Loss_Comparison', {
            'Train_Loss': train_loss,
            'Val_Loss': val_metrics['loss']
        }, epoch)
        
        # Adiciona o gráfico comparativo das acurácias
        writer.add_scalars(f'{model_type}/Accuracy_Comparison', {
            'Train_Global': train_acc_global,
            'Train_Ponderada': train_acc_pond,
            'Val_Global': val_metrics['acc_global'],
            'Val_Ponderada': val_metrics['acc_ponderada']
        }, epoch)

        writer.add_scalars(
            f"{model_type}/Recall",
            {
                cls_name: val_metrics["recall_per_class"][idx]
                for idx, cls_name in enumerate(class_names)
            },
            epoch
        )

        writer.add_scalars(
            f"{model_type}/Precision",
            {
                cls_name: val_metrics["precision_per_class"][idx]
                for idx, cls_name in enumerate(class_names)
            },
            epoch
        )

        f1_dict = {
            cls_name: val_metrics["f1_per_class"][idx]
            for idx, cls_name in enumerate(class_names)
        }

        f1_dict["Macro"] = val_metrics["f1_macro"]

        writer.add_scalars(
            f"{model_type}/F1",
            f1_dict,
            epoch
        )

        # ROC AUC and Curve
        writer.add_scalar(
            f'{model_type}/Metrics/ROC_AUC',
            val_metrics['roc_auc'],
            epoch
        )

        if (epoch + 1) % 5 == 0:
            try:
                fig_roc = log_roc_curve_multiclass(
                    val_metrics["y_true"],
                    val_metrics["y_prob"],
                    class_names
                )

                writer.add_figure(
                    f"{model_type}/Visuals/ROC_Curve",
                    fig_roc,
                    epoch
                )

                plt.close(fig_roc)

            except Exception as e:
                print(f"Warning: erro ao registrar ROC curve: {e}")

        # PR AUC and Curve
        writer.add_scalar(
            f'{model_type}/Metrics/PR_AUC',
            val_metrics['pr_auc'],
            epoch
        )

        if (epoch + 1) % 5 == 0:
            try:

                fig_pr = log_pr_curve_multiclass(
                    val_metrics['y_true'],
                    val_metrics['y_prob'],
                    class_names
                )

                writer.add_figure(
                    f'{model_type}/Visuals/PR_Curve',
                    fig_pr,
                    epoch
                )

                plt.close(fig_pr)

            except Exception as e:
                print(
                    f"Warning: não foi possível registrar curva precision-recall no tensorboard: {e}"
                )

        # Gráficos e Imagens
        try:
            fig = _confusion_matrix_figure(val_metrics['y_true'], val_metrics['y_pred'], normalize=True)
            writer.add_figure(f'{model_type}/Visuals/Confusion_Matrix', fig, epoch)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: não foi possível registrar matriz de confusão no tensorboard: {e}")
        
        try:            
            fig_erros = plot_top_errors_multiclass(
                model=model,
                loader=val_loader,
                device=device,
                num_images=10,
                class_names=class_names,
                mean=mean,
                std=std
            )

            writer.add_figure(
                f"{model_type}/Visuals/Top_Errors",
                fig_erros,
                epoch
            )
            plt.close(fig_erros)
        except Exception as e:
            print(f"Warning: não foi possível registrar top erros no tensorboard: {e}")

        try:            
            log_gradcam_tensorboard(
                writer=writer, 
                model=model, 
                loader=val_loader, 
                device=device, 
                epoch=epoch, 
                model_type=model_type,
                num_images=4,
                is_binary_loss=False, # <-- Ajuste isso de acordo com a sua Loss
                mean=mean,
                std=std
            )
        except Exception as e:
            print(f"Warning: não foi possível registrar grad cam no tensorboard: {e}")

        # Histograma 
        y_true = val_metrics['y_true']
        y_pred = val_metrics['y_pred']
        y_prob = val_metrics['y_prob']

        confidence = y_prob.max(axis=1)

        writer.add_histogram(
            f"{model_type}/Probabilities/Confidence",
            confidence,
            epoch
        )

        correct_mask = (y_true == y_pred)

        writer.add_histogram(
            f"{model_type}/Confidence/Correct",
            confidence[correct_mask],
            epoch
        )

        writer.add_histogram(
            f"{model_type}/Confidence/Wrong",
            confidence[~correct_mask],
            epoch
        )

        for cls_idx, cls_name in enumerate(class_names):

            writer.add_histogram(
                f"{model_type}/Probabilities/{cls_name}",
                y_prob[:, cls_idx],
                epoch
            )

        if use_early_stopping and getattr(early_stopping, 'early_stop', False): # pyright: ignore[reportPossiblyUnboundVariable]
            print("Early stopping interrompendo o treino...")
            break

        # if use_early_stopping:
        #     try:
        #         best_path = early_stopping.path
        #         from pathlib import Path as _Path
        #         if _Path(best_path).exists():
        #             model.load_state_dict(torch.load(best_path, map_location=device))
        #             print(f"Melhor modelo carregado de: {best_path}")
        #         else:
        #             print(f"Aviso: checkpoint não encontrado em {best_path}")
        #     except Exception as e:
        #         print(f"Warning: falha ao carregar melhor modelo: {e}")

    writer.close()
    return history