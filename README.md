# Classificação de Espécies de Flores Iris com Redes Neurais (Keras)

## Visão Geral do Projeto

Este projeto demonstra a construção e o treinamento de Redes Neurais Artificiais (RNAs) usando a biblioteca Keras (com TensorFlow como backend) para classificar espécies de flores Iris. O conjunto de dados Iris é um clássico em aprendizado de máquina, ideal para explorar os fundamentos da classificação com redes neurais. O notebook apresenta duas implementações ligeiramente diferentes do modelo, permitindo comparar os efeitos de algumas escolhas arquitetônicas e de treinamento.

## Objetivo

O objetivo principal é classificar uma flor Iris em uma das três espécies (Setosa, Versicolor, Virginica) com base em quatro características botânicas: comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala.

## Tecnologias Utilizadas

*   **Python**: Linguagem de programação principal.
*   **Keras**: API de alto nível para construção e treinamento de modelos de aprendizado de máquina.
*   **TensorFlow**: Biblioteca de código aberto para aprendizado de máquina, utilizada como backend para Keras.
*   **scikit-learn**: Biblioteca para ferramentas de aprendizado de máquina, utilizada para carregamento de dados (`load_iris`), divisão de dados (`train_test_split`), e pré-processamento (`StandardScaler`, `OneHotEncoder`).
*   **NumPy**: Biblioteca para computação numérica eficiente.
*   **Matplotlib**: Biblioteca para visualização de dados, usada para plotar a acurácia do modelo.

## Estrutura do Notebook e Análise das Células

O notebook é composto por duas células de código Python, cada uma representando uma implementação completa de uma rede neural. Ambas as células seguem uma estrutura lógica de etapas, mas diferem na arquitetura e nos parâmetros de treinamento da rede neural.

### Seção Comum a Ambas as Células (Etapas 1 e 2)

Ambas as células compartilham as seguintes etapas iniciais:

1.  **Importação de Bibliotecas**: Carrega todas as bibliotecas necessárias para manipulação de dados, pré-processamento, construção e treinamento de redes neurais, e visualização.
2.  **Carregamento e Limpeza de Dados (Etapa 1)**: O dataset Iris é carregado diretamente do `scikit-learn`. Ele contém 150 amostras de flores Iris, cada uma com 4 características e um rótulo de espécie (0, 1 ou 2).
3.  **Pré-Processamento (Etapa 2)**:
    *   **One-Hot Encoding**: Os rótulos de classe numérica (`y`) são convertidos em um formato "one-hot" (vetores binários). Por exemplo, a classe `2` se torna `[0, 0, 1]`. Isso é necessário para a função de perda `categorical_crossentropy`.
    *   **Divisão Treino/Teste**: Os dados são divididos em 80% para treinamento e 20% para teste, garantindo que o modelo seja avaliado em dados que não viu durante o treinamento. `random_state=42` assegura a reprodutibilidade da divisão.
    *   **Normalização**: As características de entrada (`X`) são padronizadas usando `StandardScaler`. Este passo é crucial para redes neurais, pois ajuda na convergência do algoritmo de otimização ao garantir que todas as características tenham a mesma escala (média 0, desvio padrão 1).

### Modelos de Redes Neurais (Etapas 3, 4 e 5)

As células diferem na definição, treinamento e avaliação do modelo de Rede Neural:

#### Célula 1 (Implementação Original)

Esta célula constrói e treina uma rede neural com a seguinte configuração:

*   **Etapa 3: Construção da Rede Neural**:
    *   **Arquitetura**: Um modelo `Sequential` do Keras com 3 camadas `Dense`:
        *   Uma primeira camada oculta com 8 neurônios e função de ativação **ReLU** (`activation='relu'`). `input_shape=(4,)` define o formato da entrada.
        *   Uma segunda camada oculta com 8 neurônios e função de ativação **ReLU**.
        *   Uma camada de saída com 3 neurônios (correspondendo às 3 classes) e função de ativação **Softmax** para gerar probabilidades de classe.
    *   **Compilação**: O modelo é compilado com o otimizador `adam`, a função de perda `categorical_crossentropy` (adequada para classificação multi-classe one-hot encoded) e a métrica `accuracy`.
*   **Etapa 4: Treinamento**: O modelo é treinado por **50 épocas** (`epochs=50`) com um `batch_size=5`. `validation_split=0.1` reserva 10% dos dados de treinamento para validação, monitorando o desempenho durante o treinamento.
*   **Etapa 5: Avaliação**: O modelo é avaliado no conjunto de teste (`X_test`, `y_test`), e a `accuracy` é impressa.
*   **Visualização**: Um gráfico de linha mostra a `accuracy` de treinamento e validação ao longo das épocas, permitindo visualizar o processo de aprendizado e identificar possíveis overfitting ou underfitting.

#### Célula 2 (Implementação Modificada)

Esta célula apresenta uma variação do modelo da célula anterior, com as seguintes mudanças:

*   **Etapa 3: Construção da Rede Neural**:
    *   **Arquitetura**: O modelo `Sequential` agora tem 4 camadas `Dense`:
        *   Uma primeira camada oculta com 8 neurônios e função de ativação **Sigmoid** (`activation='sigmoid'`).
        *   Uma segunda camada oculta com 8 neurônios e função de ativação **Sigmoid**.
        *   **Uma terceira camada oculta adicional** com 16 neurônios e função de ativação **Sigmoid**.
        *   A camada de saída permanece com 3 neurônios e ativação **Softmax**.
    *   **Compilação**: Mantém as mesmas configurações de otimizador, perda e métricas.
*   **Etapa 4: Treinamento**: O modelo é treinado por **100 épocas** (`epochs=100`) com um `batch_size=1`. `validation_split=0.1` é mantido.
*   **Etapa 5: Avaliação**: Similarmente, o modelo é avaliado no conjunto de teste, e a `accuracy` é impressa.
*   **Visualização**: Um gráfico de linha exibe a `accuracy` de treinamento e validação ao longo das épocas, refletindo as novas configurações de treinamento.

## Como Executar

Para executar este notebook, siga os passos:

1.  **Ambiente**: Certifique-se de ter um ambiente Python configurado com as bibliotecas listadas instaladas. Se estiver usando o Google Colab, a maioria delas já estará disponível.
2.  **Instalar Dependências (se necessário)**:
    ```bash
    pip install numpy pandas matplotlib scikit-learn tensorflow
    ```
3.  **Abrir o Notebook**: Faça o upload ou abra este arquivo `.ipynb` em um ambiente Jupyter (como Jupyter Notebook, JupyterLab ou Google Colab).
4.  **Executar Células**: Execute as células do notebook em ordem. Você pode usar a opção 'Executar Tudo' ou executar cada célula individualmente para observar o desempenho de cada modelo.

## Resultados Esperados e Interpretação

Ao executar cada célula, você verá:

*   O formato dos dados de entrada (`(150, 4)`).
*   Mensagens indicando o início do treinamento de cada modelo.
*   A acurácia final do respectivo modelo no conjunto de teste. Para o dataset Iris, é comum obter acurácias muito altas (próximas a 100%) devido à sua simplicidade.
*   Um gráfico mostrando a evolução da acurácia de treinamento e validação para cada modelo. É interessante comparar os gráficos dos dois modelos para entender o impacto das diferentes funções de ativação (ReLU vs. Sigmoid), o número de camadas ocultas (2 vs. 3), o número de épocas e o tamanho do lote no processo de aprendizado e na generalização.

**Interpretação Comparativa:**

*   **Funções de Ativação (ReLU vs. Sigmoid)**: A ReLU (`max(0, x)`) é geralmente preferida em camadas ocultas de redes neurais mais profundas por ajudar a mitigar o problema do "gradiente evanescente" (vanishing gradient), que pode ocorrer com a Sigmoid. A Sigmoid, embora útil na camada de saída para problemas binários, pode tornar o treinamento de camadas ocultas mais lento e menos eficaz em alguns cenários.
*   **Número de Camadas Ocultas**: Adicionar uma camada oculta adicional (como na Célula 2) pode, em tese, permitir que o modelo aprenda padrões mais complexos. No entanto, para um dataset simples como o Iris, isso pode não resultar em melhorias significativas e pode até aumentar o risco de overfitting se o modelo se tornar muito complexo para os dados.
*   **Épocas e Batch Size**: Mais épocas (100 na Célula 2) significam mais oportunidades para o modelo aprender, enquanto um `batch_size` menor (1 na Célula 2) significa que os pesos do modelo são atualizados mais frequentemente (após cada amostra), o que pode levar a um treinamento mais ruidoso, mas potencialmente a um mínimo global melhor.

Este notebook serve como um excelente ponto de partida para experimentar diferentes configurações de redes neurais e observar seus efeitos no desempenho do modelo.
