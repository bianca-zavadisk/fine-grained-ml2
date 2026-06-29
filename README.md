# fine-grained-ml2

Repositório para a implementação do projeto final da matéria "Aprendizado de Máquina 2 - MD31 2026.1" do IMPA Tech.

## Visão geral do projeto
Este repositório contém o trabalho final de classificação fine‑grained aplicado ao conjunto SLICE‑3D / ISIC (conjuntos relacionados no diretório `data/`). O objetivo é comparar e avaliar arquiteturas e estratégias de treinamento para detecção/classificação de classes finas em imagens médicas/dermatológicas.

## Modelo e técnicas testadas
Foram testadas e comparadas abordagens típicas de fine‑grained classification, incluindo:
- Redes convolucionais residuais (ResNet) e arquiteturas pré‑treinadas (transfer learning).
- Estratégias de data augmentation (transformações geométricas, normalização).
- Balanceamento de classes e pesos de perda.
- Otimizadores e agendadores de taxa de aprendizado (Adam, SGD, schedulers).
- Regularização (dropout, weight decay) e early stopping.
- Experimentos e análises estão registrados em notebooks e scripts.

Consulte os notebooks com as análises e treinamentos:
- [data_analysis.ipynb](data_analysis.ipynb)
- [training_models.ipynb](training_models.ipynb)

Exemplos anteriores e referências usadas:
- [previous_proj_examples/classificacao_com_cnn.ipynb](previous_proj_examples/classificacao_com_cnn.ipynb)
- [previous_proj_examples/vae_watermark.ipynb](previous_proj_examples/vae_watermark.ipynb)

## Contexto acadêmico
Trabalho final para a disciplina Aprendizado de Máquina 2 (ML2) do IMPA Tech — MD31 2026.1.

## Estrutura do repositório (arquivos relevantes)
- [README.md](README.md)
- [pyproject.toml](pyproject.toml) — especifica dependências e versão Python (≈ 3.12).
- [requirements.txt](requirements.txt)
- [data_analysis.ipynb](data_analysis.ipynb)
- [training_models.ipynb](training_models.ipynb)
- [isic2018_dataset.py](isic2018_dataset.py)
- [ISIC2018_Task3_Training_GroundTruth.csv](ISIC2018_Task3_Training_GroundTruth.csv)
- [ISIC2018_Task3_Training_LesionGroupings.csv](ISIC2018_Task3_Training_LesionGroupings.csv)

## Como configurar o ambiente (reprodução)
1. Clonar o repositório:
```sh
git clone <repo-url>
cd fine-grained-ml2
```

2. Criar o ambiente
```sh
python -m venv env
# macOS / Linux
source env/bin/activate
# Windows (PowerShell)
env\Scripts\Activate.ps1
```

3. Intalar as dependências
```sh
pip install -r [requirements.txt](http://_vscodecontentref_/0)
pip install -e . --no-deps
```
