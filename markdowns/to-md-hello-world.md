## O *Hello, World!* de Sistemas de Recomendação

Abaixo, uma implementação básica de um sistema de recomendação não-personalizado a partir da disponibilidade de dados de avaliações.

Todos sistemas de recomendação utilizam dados categorizados como interações. Interações se configuram toda vez que um usuário do sistema interage com um item do catálogo. 

Feedback explícito, ou avaliações, são exemplos de interações que conseguimos utilizar em sistemas de recomendação.

#### 1. Importação de Bibliotecas


```python
import numpy as np
```

#### 2. Histórico de Avaliações

Usaremos dados de um histórico de avaliações para os processos subsequentes.

Para cada usuário, um array contém as avaliações daquele usuário para cada item de um catálogo:

`usuario = [item0, item1, item2, item3, ...]`

Caso um usuário não tenha avaliado um item, deixamos a avaliação como 0.


```python
NUM_USUARIOS = 6
NUM_ITENS = 6

usuario0 = [5, 3, 0, 4, 4, 0]
usuario1 = [1, 0, 0, 3, 0, 0]
usuario2 = [0, 0, 0, 1, 0, 0]
usuario3 = [4, 0, 0, 5, 0, 2]
usuario4 = [0, 0, 5, 4, 0, 1]
usuario5 = [0, 0, 0, 0, 0, 0]
```

Vamos transformar esses dados em uma matriz (um array 2D).


```python
avaliacoes = [usuario0, usuario1, usuario2, usuario3, usuario4, usuario5]

matriz_avaliacoes = np.array(avaliacoes)
```


```python
print(matriz_avaliacoes)
```

    [[5 3 0 4 4 0]
     [1 0 0 3 0 0]
     [0 0 0 1 0 0]
     [4 0 0 5 0 2]
     [0 0 5 4 0 1]
     [0 0 0 0 0 0]]


Agora, de maneira simplista, podemos recomendar o item com a melhor avaliação média.


```python
avaliacoes_medias = np.mean(matriz_avaliacoes, axis=0)
melhor_item = np.argsort(avaliacoes_medias)[::-1][0]
print(f"Recomendação de item (aquele mais bem avaliado): Item {melhor_item}")
```

    Recomendação de item (aquele mais bem avaliado): Item 3


De fato: o item 3 não somente é aquele mais bem avaliado, como também é o item mais avaliado.
