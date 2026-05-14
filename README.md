# E-commerce Recommendation System com TensorFlow.js

Este projeto é um sistema de recomendação para e-commerce que utiliza Machine Learning (TensorFlow.js) para sugerir produtos aos usuários com base em seus perfis e histórico de compras. O treinamento do modelo e as predições ocorrem inteiramente no navegador, utilizando Web Workers para garantir a performance da interface.

## 🚀 Tecnologias Utilizadas

- **TensorFlow.js**: Para criação, treinamento e execução do modelo de rede neural.
- **tfjs-vis**: Para visualização das métricas de treinamento em tempo real.
- **JavaScript (ES6+)**: Lógica principal utilizando módulos nativos.
- **Web Workers**: Para processamento de ML em segundo plano (background).
- **Bootstrap 5**: Para estilização da interface.
- **Browser-sync**: Para desenvolvimento e servidor local.

## 📁 Estrutura do Projeto

```text
├── data/               # Arquivos JSON com dados de usuários e produtos
├── src/
│   ├── controller/     # Lógica de controle (MVC) e orquestração de eventos
│   ├── events/         # Definições de eventos e constantes compartilhadas
│   ├── service/        # Serviços para busca e manipulação de dados
│   ├── view/           # Componentes de interface e templates HTML
│   └── workers/        # Lógica de Machine Learning (Model Training Worker)
├── index.html          # Ponto de entrada da aplicação
├── style.css           # Estilização customizada
└── package.json        # Dependências e scripts do projeto
```

## 🧠 Como Funciona o Modelo

O sistema utiliza uma rede neural para calcular a similaridade entre o perfil do usuário e os atributos dos produtos.

### 1. Preparação dos Dados (Feature Engineering)
Os dados dos produtos são normalizados e transformados:
- **Preço e Idade Média**: Normalizados para uma escala de 0 a 1.
- **Categoria e Cor**: Convertidos usando *One-Hot Encoding*.
- **Pesos (Weights)**: Cada característica possui um peso específico (Categoria: 0.4, Cor: 0.3, Preço: 0.2, Idade: 0.1) para influenciar a recomendação.

### 2. Arquitetura da Rede Neural
O modelo é composto por:
- Uma camada de entrada baseada nas dimensões dos produtos.
- Camadas densas (`dense`) com ativação `relu`.
- Uma camada de saída com ativação `sigmoid` que gera um "score" de recomendação.

### 3. Treinamento
O treinamento é realizado de forma assíncrona em um Web Worker, evitando o travamento da UI. É possível acompanhar o progresso (perda e acurácia) através do painel lateral (TFVisor).

## 🛠️ Como Executar

1. Certifique-se de ter o [Node.js](https://nodejs.org/) instalado.
2. Clone este repositório.
3. No terminal, dentro da pasta do projeto, instale as dependências:
   ```bash
   npm install
   ```
4. Inicie o servidor de desenvolvimento:
   ```bash
   npm start
   ```
5. O navegador abrirá automaticamente em `http://localhost:3000`.

## 📖 Instruções de Uso

1. Ao carregar a página, selecione um usuário no campo **"Select User"**.
2. O sistema exibirá o perfil do usuário e suas compras passadas.
3. Clique em **"Train Model"** para treinar a rede neural com os dados atuais.
4. Após o treinamento, clique em **"Run Recommendation"** para gerar as sugestões de produtos personalizadas.

---
Desenvolvido como exemplo de Fundamentos de IA e LLMs.
