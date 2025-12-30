# 🐶🐱 Classificação de Gatos e Cachorros com Deep Learning

Este projeto foi desenvolvido durante a graduação como parte de uma disciplina de **Aprendizado Profundo (Deep Learning)**, com o objetivo de aplicar e comparar diferentes arquiteturas de redes neurais para classificação de imagens.

O problema abordado é a clássica tarefa de **classificação binária (Gato vs Cachorro)** utilizando imagens, explorando três abordagens distintas:
- Rede **Fully Connected**
- **Rede Convolucional (CNN)**
- **Transfer Learning** com modelo pré-treinado (**ResNet50**)

Além disso, foi criada uma aplicação em **Streamlit** que permite ao usuário carregar uma imagem, obter a classificação e visualizar um **Grad-CAM**, buscando interpretar quais regiões da imagem mais influenciaram a decisão do modelo.

---

## 📌 Objetivos do Projeto

- Aplicar conceitos fundamentais de Deep Learning
- Comparar diferentes arquiteturas de redes neurais
- Utilizar modelos pré-treinados (Transfer Learning)
- Explorar interpretabilidade de modelos com Grad-CAM
- Desenvolver uma interface simples para inferência do modelo

---

## 🗂️ Estrutura do Repositório

```
├── 00_salvando_dados.ipynb
├── 01_fc_cat_dog.ipynb
├── 02_conv_cat_dog.ipynb
├── 03_resnet_cat_dog.ipynb
├── 04_streamlit_grad-cam.py
├── requirements.txt
└── README.md
```

---

## 📂 Descrição dos Arquivos

### `00_salvando_dados.ipynb`
Notebook responsável por:
- Carregar o dataset Cats vs Dogs
- Realizar pré-processamento das imagens
- Separar os dados em treino, validação e teste
- Salvar os dados para reutilização nos demais experimentos

---

### `01_fc_cat_dog.ipynb`
Implementação de um modelo **Fully Connected**, utilizado como baseline.
- Flatten das imagens
- Camadas densas
- Avaliação das limitações desse tipo de abordagem para imagens

---

### `02_conv_cat_dog.ipynb`
Implementação de uma **Rede Neural Convolucional (CNN)** construída do zero.
- Camadas convolucionais
- Pooling
- Dropout
- Melhor desempenho em relação ao modelo fully connected

---

### `03_resnet_cat_dog.ipynb`
Uso de **Transfer Learning** com **ResNet50**.
- Modelo pré-treinado
- Substituição da camada final
- Fine-tuning
- Melhor desempenho geral entre os modelos testados

---

### `04_streamlit_grad-cam.py`
Aplicação em **Streamlit** que:
- Permite upload de uma imagem
- Classifica como **Gato** ou **Cachorro**
- Gera um mapa de ativação **Grad-CAM**
- Tenta destacar quais regiões da imagem influenciaram a decisão do modelo

> ⚠️ Observação: a implementação do Grad-CAM ainda pode ser aprimorada e não apresenta resultados ideais em todos os casos.

---

## 🧠 Grad-CAM

O Grad-CAM foi utilizado como uma tentativa de interpretação das decisões do modelo baseado em ResNet50.  
Apesar de funcional, a implementação ainda apresenta limitações e serve como base para estudos futuros sobre interpretabilidade de modelos de Deep Learning.

---

## 🚀 Como Executar o Projeto

### 1️⃣ Instalar dependências

pip install -r requirements.txt

### 2️⃣ Baixar e ajustar o dataset

Acessar o link: https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification/data

Fazer o download dos arquivos e colocá-los na mesma pasta dos códigos

Lembre-se de manter os arquivos com o mesmo nome:

**Pasta com as imagens**: cat_dog
**CSV com classificação**: cat_dog.csv

### 3️⃣ Treinar os modelos

Execute os notebooks na seguinte ordem:

00_salvando_dados.ipynb

01_fc_cat_dog.ipynb

02_conv_cat_dog.ipynb

03_resnet_cat_dog.ipynb

### 4️⃣ Executar a aplicação Streamlit

streamlit run 04_streamlit_grad-cam.py

---

## 📊 Resultados

De forma geral:

O modelo Fully Connected apresentou desempenho inferior

A CNN apresentou melhorias significativas

O modelo ResNet50 obteve o melhor desempenho em termos de acurácia e generalização

---

## 📚 Dataset

O dataset utilizado é o clássico Cats vs Dogs, disponível publicamente no Kaggle (https://www.kaggle.com/datasets/ashfakyeafi/cat-dog-images-for-classification/data).

O dataset não está incluído neste repositório devido ao seu tamanho.

## 🗃️ Modelos

Os códigos vão gerar os seguintes arquivos com os modelos treinados:

**Fully Connected**: fc_melhor_modelo.pth
**CNN**: melhor_modelo_conv.pth
**ResNet**: melhor_modelo_resnet.pth

Não troque o nome dos arquivos para que não de nenhum erro nos códigos.

## 🔮 Trabalhos Futuros

Melhorar a implementação do Grad-CAM

Testar outros modelos pré-treinados

Ajustar hiperparâmetros

Avaliar métricas adicionais além da acurácia

## 👤 Autor

Projeto desenvolvido por Felipe Relvas