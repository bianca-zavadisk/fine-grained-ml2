'''
Visualizações da classificação binária
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

def exibir_imagens_por_classe(loader, num_samples=3):
    """Exibe um número específico de imagens para cada classe presente no DataLoader.

    :param loader: PyTorch DataLoader
    :param num_samples: Quantidade de imagens a exibir por classe
    """
    # Tenta pegar os nomes das classes do dataset, se existirem
    class_names = (
        loader.dataset.classes if hasattr(loader.dataset, "classes") else None
    )

    imagens_por_classe = {}

    # 1. Coleta as imagens necessárias
    for num_batch, (images, labels) in enumerate(loader):
        for img, lbl in zip(images, labels):
            classe = lbl.item() if isinstance(lbl, torch.Tensor) else lbl

            if classe not in imagens_por_classe:
                imagens_por_classe[classe] = []

            # Adiciona a imagem se ainda não atingiu o limite para essa classe
            if len(imagens_por_classe[classe]) < num_samples:
                imagens_por_classe[classe].append(img)

        # Verifica se já temos amostras suficientes de TODAS as classes coletadas até agora
        # (Isso evita rodar o loader inteiro se o dataset for gigante)
        if class_names and len(imagens_por_classe) == len(class_names):
            if all(
                len(imgs) == num_samples for imgs in imagens_por_classe.values()
            ):
                break

    classes_ordenadas = sorted(imagens_por_classe.keys())
    num_classes = len(classes_ordenadas)

    fig, axes = plt.subplots(
        num_classes, num_samples, figsize=(num_samples * 2, num_classes * 2)
    )

    # Ajuste para o caso de termos apenas 1 classe ou 1 amostra (evita quebra do numpy)
    if num_classes == 1 and num_samples == 1:
        axes = np.array([[axes]])
    elif num_classes == 1:
        axes = np.expand_dims(axes, axis=0)
    elif num_samples == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx_classe, classe in enumerate(classes_ordenadas):
        imagens = imagens_por_classe[classe]

        for idx_img in range(num_samples):
            ax = axes[idx_classe, idx_img]

            if idx_img >= len(imagens):
                ax.axis("off")
                continue

            img = imagens[idx_img]

            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).cpu().numpy()    # (C, H, W) -> (H, W, C)

            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap="gray")
            else:
                ax.imshow(img)

            ax.axis("off")

            if idx_img == 0:
                nome_classe = (
                    class_names[classe] if class_names else f"Classe {classe}"
                )
                ax.set_title(
                    nome_classe, loc="left", fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

def plot_roc(model, loader, device, model_name='', type='binary', num_classes=None, student_run_tag=None, output_dir='finalProject_outputs'):
    """
    Plota a curva ROC para o modelo.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_true = []
    y_scores = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            # mover tensores para o mesmo device do modelo
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            # calcula probabilidades compatíveis com binário/multiclass
            if outputs.dim() == 1 or outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs.view(-1)).cpu().numpy()
                y_scores.extend(probs)
            else:
                probs_all = torch.softmax(outputs, dim=1)
                if outputs.shape[1] == 2:
                    probs = probs_all[:, 1].cpu().numpy()
                    y_scores.extend(probs)
                else:
                    # multiclass: armazena todas as probabilidades (por classe)
                    # aqui acumulamos por batch; converteremos depois para array NxC
                    y_scores.extend(probs_all.cpu().numpy())

            y_true.extend(labels.cpu().numpy())

            del outputs, images, labels
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    import numpy as np

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if type == 'multiclass':
        n_classes = num_classes
        # espera y_scores shape (N, C)
        if y_scores.ndim == 1:
            raise RuntimeError("Esperado scores multiclass NxC, mas recebeu vetor unidimensional.")
        y_test_bin = label_binarize(y_true, classes=np.arange(n_classes))
        fpr = dict(); tpr = dict(); roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i]) # type: ignore
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_scores.ravel()) # type: ignore
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=4)
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} ROC (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Multi-class One-vs-Rest (OvR) ROC Curve'); plt.legend(loc="lower right"); plt.grid(True)
        
        save_dir = f'./{output_dir}/{student_run_tag}'
        file_path = f'{save_dir}/roc_curve_{model_name}.png'
        plt.savefig(file_path)
        
        plt.show()

    else:
        # binário: y_scores deve ser vetor com prob da classe positiva
        if y_scores.ndim > 1:
            # se acumulamos matrizes por batch, podem precisar pegar coluna 1
            if y_scores.shape[1] == 2:
                y_scores = y_scores[:, 1]
            else:
                raise RuntimeError("Scores com dimensão inesperada para ROC binário.")
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)'); plt.legend(loc="lower right"); plt.show()
        
        save_dir = f'./{output_dir}/{student_run_tag}'
        file_path = f'{save_dir}/roc_curve_{model_name}.png'
        plt.savefig(file_path)
        
        plt.show()
        
def plot_eval(history, model_name, student_run_tag='', output_dir='finalProject_outputs'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    metrics = [('loss', 'Loss'), ('acc', 'Accuracy')]

    for i, (key, label) in enumerate(metrics):
        ax = axes[i]

        ax.plot(history[f'train_{key}'], label=f'Train {label}', color='darkorchid', lw=2)
        ax.plot(history[f'val_{key}'], label=f'Validation {label}', color='seagreen', lw=2)

        ax.set_title(f'Training and Validation {label} ({model_name.upper()})')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    save_dir = f'./{output_dir}/{student_run_tag}'
    file_path = f'{save_dir}/evaluation_curves_{model_name}.png'
    plt.savefig(file_path)
    plt.show()
    print(f'Painel de avaliação salvo em: {file_path}')
    
def plot_confusion_matrix(model, loader, device, model_name='', classes=None, normalize=False, figsize=(8,6), cmap='Blues', class_map=None, num_classes=None, student_run_tag='', output_dir='finalProject_outputs', log=True):
    """
    Calcula e plota a matriz de confusão para `model` sobre `loader`.
    ...
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            # Decide predicões a partir da saída do modelo
            if outputs.dim() == 1 or outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs.view(-1))
                preds = (probs > 0.5).long().cpu().numpy()
            else:
                preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.append(preds)
            y_true.append(labels.cpu().numpy())

            del outputs, images, labels
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Decide classes e rótulos a mostrar nos eixos
    if class_map is not None:
        # Normaliza class_map para inv_map: int_index -> name_str
        inv_map = {}
        for k, v in class_map.items():
            # tenta tratar chave como índice (int -> name)
            try:
                idx = int(k)
                inv_map[idx] = str(v)
                continue
            except Exception:
                pass
            # tenta tratar valor como índice (name -> int)
            try:
                idx = int(v)
                inv_map[idx] = str(k)
                continue
            except Exception:
                pass
            raise ValueError("class_map deve mapear int->str (ex: {0:'benigno'}) ou str->int (ex: {'benigno':0}).")
        ordered_indices = sorted(inv_map.keys())
        if classes is None:
            classes = ordered_indices
        # tick labels na mesma ordem das classes
        tick_labels = [inv_map.get(c, str(c)) for c in classes]
    else:
        # sem class_map: comportamento anterior
        if classes is None:
            try:
                classes = list(range(num_classes))
            except Exception:
                classes = np.unique(np.concatenate([y_true, y_pred])).tolist()
        # se classes são ints, usa-os como rótulos simples
        tick_labels = [str(c) for c in classes]

    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true' if normalize else None)

    plt.figure(figsize=figsize)
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix' + (' (normalized)' if normalize else '') + f' - {model_name}')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, tick_labels, rotation=45)
    plt.yticks(tick_marks, tick_labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if normalize:
                s = f"{val:.2f}"
            else:
                s = f"{int(val)}"
            plt.text(j, i, s,
                     horizontalalignment="center",
                     color="white" if val > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if not log:
        save_dir = f'./{output_dir}/{student_run_tag}'
        file_path = f'{save_dir}/confusion_matrix_{model_name}.png'
        plt.savefig(file_path)
    
    plt.show()

    return cm

def plot_top_errors(model, loader, device, num_images=5, class_names=['Benigno', 'Maligno'], mean=None, std=None, log=True, student_run_tag='', output_dir='finalProject_outputs', model_name=''):
    """
    Percorre o DataLoader, identifica os piores erros da rede e retorna uma figura Matplotlib.
    """
    model.eval()
    
    # Armazenar erros: (confiança_no_erro, imagem_cpu, probabilidade_maligno)
    false_positives = [] 
    false_negatives = [] 
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            
            if outputs.dim() == 1 or outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs.view(-1))
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
            preds = (probs > 0.5).long()
            
            # Avalia cada imagem do batch
            for i in range(len(labels)):
                true_y = labels[i].item()
                pred_y = preds[i].item()
                prob_maligno = probs[i].item()
                
                if true_y == 0 and pred_y == 1:
                    # FP: Era Benigno. O quão grave é o erro? Quanto mais perto de 1.0 (100%), pior.
                    false_positives.append((prob_maligno, inputs[i].detach().cpu(), prob_maligno))
                    
                elif true_y == 1 and pred_y == 0:
                    # FN: Era Maligno. O quão grave é o erro? Quanto mais perto de 0.0, pior.
                    # Usamos (1 - prob_maligno) para que erros piores tenham valores maiores, facilitando a ordenação.
                    false_negatives.append((1 - prob_maligno, inputs[i].detach().cpu(), prob_maligno))
                    
    # Ordena de forma decrescente pela gravidade do erro
    false_positives.sort(key=lambda x: x[0], reverse=True)
    false_negatives.sort(key=lambda x: x[0], reverse=True)
    
    # Pega apenas o top K solicitado
    top_fp = false_positives[:num_images]
    top_fn = false_negatives[:num_images]
    
    # Prepara a figura
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=1) # Ajuste para num_images=1 não quebrar os índices
        
    def imshow(img_tensor, ax, title):
        # Converte de tensor (C, H, W) para numpy (H, W, C)
        img = img_tensor.numpy().transpose((1, 2, 0))

        img = std * img + mean

        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(title, fontsize=10, color='darkred')
        ax.axis('off')

    # Plotando os Falsos Positivos (Linha Superior)
    axes[0, 0].set_ylabel('Falsos Positivos', fontsize=12, weight='bold') # Label da linha
    for i in range(num_images):
        if i < len(top_fp):
            _, img, prob = top_fp[i]
            imshow(img, axes[0, i], f"FP (Real: {class_names[0]})\nPreviu Maligno com {prob:.1%}")
        else:
            axes[0, i].axis('off')
            
    # Plotando os Falsos Negativos (Linha Inferior)
    axes[1, 0].set_ylabel('Falsos Negativos', fontsize=12, weight='bold') # Label da linha
    for i in range(num_images):
        if i < len(top_fn):
            _, img, prob = top_fn[i]
            # Probabilidade próxima de 0, significa que o modelo estava confiante que era benigno
            imshow(img, axes[1, i], f"FN (Real: {class_names[1]})\nPreviu Benigno com {prob:.1%}")
        else:
            axes[1, i].axis('off')

    plt.tight_layout()
    
    if not log:
        save_dir = f'./{output_dir}/{student_run_tag}'
        file_path = f'{save_dir}/confusion_matrix_{model_name}.png'
        plt.savefig(file_path)
    
        plt.show()
    
    return fig

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def plot_gradcam(model, loader, device, num_images=4, is_binary_loss=False, mean=None, std=None, student_run_tag='', output_dir='finalProject_outputs', model_name=''):
    model.eval()
    
    # 1. Define a camada alvo da ResNet
    target_layers = [model.layer4[-1]]
    
    # 2. Inicializa o GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Define o alvo: Queremos ver o que faz a rede pensar que é "Maligno" (Classe 1)
    if is_binary_loss: # Se estiver usando BCEWithLogitsLoss (saída de 1 dimensão)
        targets = [BinaryClassifierOutputTarget(1)] * num_images
    else:              # Se estiver usando CrossEntropyLoss (saída de 2 dimensões)
        targets = [ClassifierOutputTarget(1)] * num_images

    # Pega um batch de validação
    inputs, labels = next(iter(loader))
    inputs = inputs[:num_images].to(device)
    labels = labels[:num_images]

    # 3. Gera os mapas de ativação (retorna um array numpy)
    grayscale_cams = cam(input_tensor=inputs, targets=targets) # type: ignore

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        # 4. Desnormaliza a imagem do PyTorch para exibição RGB
        
        img_tensor = inputs[i].detach().cpu().numpy().transpose((1, 2, 0))
        img_rgb = std * img_tensor + mean
        img_rgb = np.clip(img_rgb, 0, 1) # Garante que os pixels fiquem entre 0 e 1

        # 5. Sobrepõe o mapa de calor na imagem original
        cam_image = show_cam_on_image(img_rgb, grayscale_cams[i, :], use_rgb=True)

        true_label = "Maligno" if labels[i].item() == 1 else "Benigno"
        
        axes[i].imshow(cam_image)
        axes[i].set_title(f"Real: {true_label}\nMapa de Ativação (Maligno)")
        axes[i].axis('off')

    plt.tight_layout()
    
    save_dir = f'./{output_dir}/{student_run_tag}'
    file_path = f'{save_dir}/confusion_matrix_{model_name}.png'
    plt.savefig(file_path)
        
    plt.show()
    
    # Importante: limpar memória da GPU ocupada pelos hooks do GradCAM
    del cam
    
def log_gradcam_tensorboard(writer, model, loader, device, epoch, model_type="model", num_images=4, is_binary_loss=False, mean=None, std=None):
    model.eval()
    target_layers = [model.layer4[-1]]
    
    # O GradCAM adiciona 'hooks' ao modelo. Usar with garante que eles sejam limpos depois.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        
        if is_binary_loss:
            targets = [BinaryClassifierOutputTarget(1)] * num_images
        else:
            targets = [ClassifierOutputTarget(1)] * num_images

        inputs, labels = next(iter(loader))
        inputs = inputs[:num_images].to(device)
        
        grayscale_cams = cam(input_tensor=inputs, targets=targets) # type: ignore
        
        cam_images_tensor = []
        
        for i in range(num_images):
            # Desnormalização
            img_tensor = inputs[i].detach().cpu().numpy().transpose((1, 2, 0))
            img_rgb = std * img_tensor + mean
            img_rgb = np.clip(img_rgb, 0, 1)

            # Sobreposição
            cam_image = show_cam_on_image(img_rgb, grayscale_cams[i, :], use_rgb=True)
            
            # Converte de volta para Tensor (C, H, W) para o TensorBoard e normaliza para [0, 1]
            cam_image_tensor = torch.from_numpy(cam_image).permute(2, 0, 1).float() / 255.0
            cam_images_tensor.append(cam_image_tensor)
            
        # Cria um grid com as imagens lado a lado
        grid = torchvision.utils.make_grid(cam_images_tensor, nrow=num_images)
        writer.add_image(f'{model_type}/Visuals/GradCAM_Maligno', grid, epoch)
        
from sklearn.metrics import roc_curve, auc

def log_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr,
        tpr,
        label=f'ROC AUC = {roc_auc:.4f}'
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        alpha=0.7,
        label='Random'
    )

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig