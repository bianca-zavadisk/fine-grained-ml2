'''
FUNÇÕES DE TREINAMENTO
'''

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model): # Removeu val_acc
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            return

        # Verifica se a loss de validação diminuiu significativamente
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0  # Reseta o contador se melhorou
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} de {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        '''Salva o modelo quando há melhora na loss de validação.'''
        import torch
        torch.save(model.state_dict(), self.path)
        
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx=None, writer=None, model_type="model"):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    desc = f"Epoch {epoch_idx}" if epoch_idx else "Train"
    loop = tqdm(loader, desc=desc, leave=False)

    for batch_idx, (inputs, labels) in enumerate(loop, start=1):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)

        predicted_cpu = predicted.cpu()
        labels_cpu = labels.cpu()

        total += labels_cpu.size(0)
        correct += predicted_cpu.eq(labels_cpu).sum().item()

        current_loss = total_loss / total
        current_acc = correct / total
        loop.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

        if writer is not None:
            global_step = (epoch_idx - 1) * len(loader) + batch_idx if epoch_idx else batch_idx
            writer.add_scalar(f'{model_type}/batch_loss', loss.item(), global_step)
            if batch_idx == 1:
                try:
                    imgs = inputs.detach().cpu()[:4]
                    writer.add_images(f'{model_type}/examples', imgs, epoch_idx)
                except Exception:
                    pass

        del outputs, loss, predicted, inputs, labels, predicted_cpu, labels_cpu
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        
        _, predicted = outputs.max(1)
        predicted = predicted.cpu()
        labels_cpu = labels.cpu()

        total += labels_cpu.size(0)
        correct += predicted.eq(labels_cpu).sum().item()

        del outputs, loss, predicted, labels, inputs, labels_cpu
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, model_type="", use_early_stopping = True, patience=5, log_every=5, student_run_tag='', output_dir='finalProject_outputs'):
    from torch.utils.tensorboard import SummaryWriter 
    writer = SummaryWriter(log_dir=f'./{output_dir}/{student_run_tag}/runs/{model_type}')

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    if use_early_stopping:
        early_stopping = EarlyStopping(patience=patience, path=f'./{output_dir}/{student_run_tag}/best_model_{model_type}.pth')

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

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch_idx=epoch+1, writer=writer, model_type=model_type
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if ((epoch + 1) % log_every == 0) or (epoch + 1 == epochs) or (epoch + 1 == 1):
            print(f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if use_early_stopping:
            early_stopping(val_loss, model) # type: ignore

        writer.add_scalar(f'{model_type}/train_loss', train_loss, epoch) 
        writer.add_scalar(f'{model_type}/train_acc', train_acc, epoch) 
        writer.add_scalar(f'{model_type}/val_loss', val_loss, epoch) 
        writer.add_scalar(f'{model_type}/val_acc', val_acc, epoch)

        # calcula e registra matriz de confusão da validação no tensorboard
        try:
            import numpy as np
            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)

                    if outputs.dim() == 1 or outputs.shape[1] == 1:
                        probs = torch.sigmoid(outputs.view(-1))
                        preds = (probs > 0.5).long().cpu().numpy()
                    else:
                        preds = outputs.argmax(dim=1).cpu().numpy()

                    y_pred.append(preds)
                    y_true.append(labels.cpu().numpy())

                    del outputs, inputs, labels
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

            if len(y_true) > 0:
                y_true = np.concatenate(y_true)
                y_pred = np.concatenate(y_pred)
                fig = _confusion_matrix_figure(y_true, y_pred, normalize=True)
                writer.add_figure(f'{model_type}/confusion_matrix', fig, epoch)
                import matplotlib.pyplot as plt
                plt.close(fig)
        except Exception as e:
            print(f"Warning: não foi possível registrar matriz de confusão no tensorboard: {e}")

        if use_early_stopping and early_stopping.early_stop: # type: ignore
            print("Early stopping interrompendo o treino...")
            break

    # Carrega o melhor estado do modelo antes de retornar
    # if use_early_stopping:
    #     model.load_state_dict(torch.load(f'./{output_dir}/{student_run_tag}/best_model_{model_type}.pth'))

    writer.close()

    return history

'''
FUNÇÕES DE AVALIAÇÃO
'''

def evaluate_classification(model, loader, device=device):
    """
    Avalia o modelo sobre o loader e computa métricas agregadas:
    accuracy, balanced_accuracy, precision/recall/f1 por classe, confusion matrix e ROC AUC (quando aplicável).
    Retorna um dict com as métricas.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            # trata casos binário (1 saída ou 2 saídas) e multiclass (>=2 saídas)
            if outputs.dim() == 1 or outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs.view(-1))
                preds = (probs > 0.5).long()
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
            else:
                probs_all = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                # para problema binário multiclass-style, pega prob da classe 1
                if outputs.shape[1] == 2:
                    all_probs.append(probs_all[:, 1].cpu())
                else:
                    all_probs.append(probs_all.cpu())  # multiclass scores, usado apenas se necessário
                all_preds.append(preds.cpu())

            all_labels.append(labels.cpu())

            del outputs, images, labels
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    import numpy as np
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_recall_fscore_support,
                                 confusion_matrix, roc_auc_score, classification_report)

    y_true = np.concatenate([t.numpy() for t in all_labels])
    y_pred = np.concatenate([t.numpy() for t in all_preds])

    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))

    # per-class precision/recall/f1 and support
    precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class = {}
    classes = np.unique(np.concatenate([y_true, y_pred]))
    for i, cls in enumerate(classes):
        per_class[int(cls)] = {
            'precision': float(precisions[i]), # type: ignore
            'recall': float(recalls[i]), # type: ignore
            'f1': float(f1s[i]), # type: ignore
            'support': int(supports[i]) # type: ignore
        }
    metrics['per_class'] = per_class

    # aggregated macro/binary metrics
    avg = 'binary' if len(classes) == 2 else 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
    metrics.update({'precision': float(precision), 'recall': float(recall), 'f1': float(f1)})

    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # tenta ROC AUC quando for binário e tivermos scores adequados
    try:
        if len(all_probs) > 0:
            probs_cat = np.concatenate([p.numpy() for p in all_probs])
            if probs_cat.ndim == 1:
                metrics['roc_auc'] = float(roc_auc_score(y_true, probs_cat))
            elif probs_cat.ndim == 2 and probs_cat.shape[1] == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, probs_cat[:, 1]))
            else:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
    except Exception:
        metrics['roc_auc'] = None

    # relatório resumido
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision ({avg}): {metrics['precision']:.4f}")
    print(f"Recall ({avg}): {metrics['recall']:.4f}")
    print(f"F1 ({avg}): {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics['confusion_matrix']))
    print("Per-class metrics:")
    for cls, m in metrics['per_class'].items():
        print(f" Class {cls}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}, support={m['support']}")

    return metrics