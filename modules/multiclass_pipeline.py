import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import modules.visualizations as visu

class GenericEarlyStopping:
    def __init__(
        self,
        patience=5,
        min_delta=0.0,
        mode='max',  # 'max' para AUC, F1; 'min' para loss
        path='checkpoint.pt'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score, model):

        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
            return

        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx=None, writer=None, model_type="model"):
    model.train()

    total_loss = 0.0
    
    # Contadores globais
    correct_global = 0
    total_global = 0
    
    # Contadores por classe (0 = Benigno, 1 = Maligno)
    correct_0 = 0
    total_0 = 0
    correct_1 = 0
    total_1 = 0

    desc = f"Epoch {epoch_idx}" if epoch_idx else "Train"
    loop = tqdm(loader, desc=desc, total=len(loader), unit='batch', leave=False, dynamic_ncols=True)

    for batch_idx, (inputs, labels) in enumerate(loop, start=1):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        
        if outputs.dim() == 1 or outputs.shape[1] == 1:
            probs = torch.sigmoid(outputs.view(-1))
            predicted = (probs > 0.5).long()
        else:
            _, predicted = outputs.max(1)

        # Usamos view(-1) para garantir tensores 1D, facilitando a lógica abaixo
        predicted_cpu = predicted.cpu().view(-1)
        labels_cpu = labels.cpu().view(-1)

        # --- 1. Acurácia Global ---
        total_global += labels_cpu.size(0)
        correct_global += predicted_cpu.eq(labels_cpu).sum().item()

        # --- 2. Acurácia Ponderada (Otimizada para o loop) ---
        mask_0 = (labels_cpu == 0)
        mask_1 = (labels_cpu == 1)
        
        total_0 += mask_0.sum().item()
        total_1 += mask_1.sum().item()
        
        correct_0 += (predicted_cpu[mask_0] == 0).sum().item()
        correct_1 += (predicted_cpu[mask_1] == 1).sum().item()
        
        # Proteção contra divisão por zero nos primeiros batches
        recall_0 = correct_0 / total_0 if total_0 > 0 else 0.0
        recall_1 = correct_1 / total_1 if total_1 > 0 else 0.0
        
        current_loss = total_loss / total_global
        current_acc_global = correct_global / total_global
        current_acc_pond = (recall_0 + recall_1) / 2.0

        # --- 3. Atualiza a barra de progresso ---
        loop.set_postfix(
            loss=f"{current_loss:.4f}", 
            acc=f"{current_acc_global:.4f}", 
            acc_pond=f"{current_acc_pond:.4f}"
        )

        if writer is not None:
            global_step = (epoch_idx - 1) * len(loader) + batch_idx if epoch_idx else batch_idx
            writer.add_scalar(f'{model_type}/batch_loss', loss.item(), global_step)
            if batch_idx == 1 and epoch_idx == 1:
                try:
                    imgs = inputs.detach().cpu()[:4]
                    writer.add_images(f'{model_type}/examples', imgs, epoch_idx)
                except Exception:
                    pass

        del outputs, loss, predicted, inputs, labels, predicted_cpu, labels_cpu, mask_0, mask_1
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader.dataset)
    final_acc_global = correct_global / total_global
    final_acc_pond = (recall_0 + recall_1) / 2.0 # pyright: ignore[reportPossiblyUnboundVariable, reportOperatorIssue]
    
    # Agora a função retorna também a acurácia ponderada de treino final
    return avg_loss, final_acc_global, final_acc_pond

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        
        # Extrai probabilidades para a classe positiva (maligno)
        if outputs.dim() == 1 or outputs.shape[1] == 1:
            probs = torch.sigmoid(outputs.view(-1))
            preds = (probs > 0.5).long()
            probs_pos = probs
        else:
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            probs_pos = probs[:, 1]

        all_targets.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs_pos.cpu().numpy())

        del outputs, loss, inputs, labels
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader.dataset)
    
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = 0.0
    
    # Gera todas as métricas necessárias para dados desbalanceados
    metrics = {
        'loss': avg_loss,
        'acc_global': (y_true == y_pred).mean(),
        'acc_ponderada': balanced_accuracy_score(y_true, y_pred),

        # Gerais
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),

        # Por classe
        'f1_benigno': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        'precision_benigno': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'recall_benigno': recall_score(y_true, y_pred, pos_label=0, zero_division=0),

        'f1_maligno': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'precision_maligno': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_maligno': recall_score(y_true, y_pred, pos_label=1, zero_division=0),

        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,

        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
    }
    return metrics

def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, model_type="", use_early_stopping=True, patience=5, log_every=5,student_run_tag='', output_dir='finalProject_outputs'):
    from torch.utils.tensorboard import SummaryWriter 
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

    for epoch in range(epochs):
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
            print(f"Epoch {epoch+1}/{epochs}: "
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
            f'{model_type}/F1',
            {
                'Geral': val_metrics['f1_macro'],
                'Benigno': val_metrics['f1_benigno'],
                'Maligno': val_metrics['f1_maligno']
            },
            epoch
        )

        writer.add_scalars(
            f'{model_type}/Precision',
            {
                'Geral': val_metrics['precision_macro'],
                'Benigno': val_metrics['precision_benigno'],
                'Maligno': val_metrics['precision_maligno']
            },
            epoch
        )

        writer.add_scalars(
            f'{model_type}/Recall',
            {
                'Geral': val_metrics['recall_macro'],
                'Benigno': val_metrics['recall_benigno'],
                'Maligno': val_metrics['recall_maligno']
            },
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
                fig_roc = visu.log_roc_curve(
                    val_metrics['y_true'],
                    val_metrics['y_prob']
                )

                writer.add_figure(
                    f'{model_type}/Visuals/ROC_Curve',
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
                writer.add_pr_curve(
                    f'{model_type}/Visuals/PR_Curve',
                    torch.tensor(val_metrics['y_true']),
                    torch.tensor(val_metrics['y_prob']),
                    global_step=epoch
                )
            except Exception as e:
                print(f"Warning: não foi possível registrar curva precision-recall no tensorboard: {e}")

        # Gráficos e Imagens
        try:
            fig = _confusion_matrix_figure(val_metrics['y_true'], val_metrics['y_pred'], normalize=True)
            writer.add_figure(f'{model_type}/Visuals/Confusion_Matrix', fig, epoch)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: não foi possível registrar matriz de confusão no tensorboard: {e}")
        
        try:            
            fig_erros = visu.plot_top_errors(model, val_loader, device, num_images=5)
            writer.add_figure(f'{model_type}/Visuals/Top_Errors', fig_erros, epoch)
            plt.close(fig_erros)
        except Exception as e:
            print(f"Warning: não foi possível registrar top erros no tensorboard: {e}")

        try:            
            visu.log_gradcam_tensorboard(
                writer=writer, 
                model=model, 
                loader=val_loader, 
                device=device, 
                epoch=epoch, 
                model_type=model_type,
                num_images=4,
                is_binary_loss=False # <-- Ajuste isso de acordo com a sua Loss
            )
        except Exception as e:
            print(f"Warning: não foi possível registrar grad cam no tensorboard: {e}")

        # Histograma  
        y_true = val_metrics['y_true']
        y_prob = val_metrics['y_prob']

        # Benigno  -> concentrado próximo de 0
        writer.add_histogram(
            f'{model_type}/Probabilities/Benigno',
            y_prob[y_true == 0],
            epoch
        )

        # Maligno  -> concentrado próximo de 1
        writer.add_histogram(
            f'{model_type}/Probabilities/Maligno',
            y_prob[y_true == 1],
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