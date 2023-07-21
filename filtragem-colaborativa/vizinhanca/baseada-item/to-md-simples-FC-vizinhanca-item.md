### Implementação Básica de Filtragem Colaborativa por Vizinhança baseada em Itens

Filtragem colaborativa é um processo oriundo de sistemas de recomendação, onde o objetivo é filtrar uma lista de itens para sugerir a um usuário o item mais "relevante" dessa lista.

Itens são avaliados como "relevante" a partir de uma análise que busca por similaridades em um histórico de avaliações.

Abaixo, uma implementação básica de filtragem colaborativa por vizinhança baseada em itens foi conduzida.

#### 1. Importação de Bibliotecas


```python
import numpy as np
```

#### 2. Histórico de Avaliações

Usaremos dados de um histórico de avaliações para os processos subsequentes.

Para cada usuário, um array contém as avaliações daquele usuário para cada item de um catálogo:

`usuario = [item0, item1, item2, item3, ...]`

Caso um usuário não tenha avaliado um item, deixamos a avaliação como `0`.


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

    OUTPUT
    [[5 3 0 4 4 0]
     [1 0 0 3 0 0]
     [0 0 0 1 0 0]
     [4 0 0 5 0 2]
     [0 0 5 4 0 1]
     [0 0 0 0 0 0]]


Iremos transpor nossa matriz de avaliações, de modo que linhas serão itens, e colunas serão usuários.


```python
matriz_itens = np.transpose(matriz_avaliacoes)
```


```python
print(matriz_itens)
```

    OUTPUT
    [[5 1 0 4 0 0]
     [3 0 0 0 0 0]
     [0 0 0 0 5 0]
     [4 3 1 5 4 0]
     [4 0 0 0 0 0]
     [0 0 0 2 1 0]]


Isso é necessário. Agora, cada linha contém um vetor de todas avaliações dadas a um item.

Podemos comparar esse vetores para ver o quão similar cada item é.

#### 3. Calculando a similaridade entre dois itens.

Vamos definir uma função que pega dois itens e retorna a similaridade entre eles.


```python
def calcular_similaridade(vetor1: np.ndarray, vetor2: np.ndarray) -> float:
    """
    Calcula a similaridade entre dois vetores a partir do cosseno do ângulo entre
    eles.

    Args:
        item1: numpy.ndarray do primeiro vetor
        item2: numpy.ndarray do segundo vetor

    Returns:
        similaridade: float correspondendo a similaridade entre os dois vetores
    """

    dot = np.dot(vetor1, vetor2)
    
    magnitude1 = np.linalg.norm(vetor1)
    magnitude2 = np.linalg.norm(vetor2)

    similaridade = dot / (magnitude1 * magnitude2)

    return similaridade
```

Com esses materiais, já conseguimos calcular a similaridade entre itens.


```python
for i, item in enumerate(matriz_itens):
    for j, outro_item in enumerate(matriz_itens):
        sim = calcular_similaridade(item, outro_item)
        print(f"Similaridade entre Itens {i} e {j}: {sim:.3g}")
```

    OUTPUT
    Similaridade entre Itens 0 e 0: 1
    Similaridade entre Itens 0 e 1: 0.772
    Similaridade entre Itens 0 e 2: 0
    Similaridade entre Itens 0 e 3: 0.811
    Similaridade entre Itens 0 e 4: 0.772
    Similaridade entre Itens 0 e 5: 0.552
    Similaridade entre Itens 1 e 0: 0.772
    Similaridade entre Itens 1 e 1: 1
    Similaridade entre Itens 1 e 2: 0
    Similaridade entre Itens 1 e 3: 0.489
    Similaridade entre Itens 1 e 4: 1
    Similaridade entre Itens 1 e 5: 0
    Similaridade entre Itens 2 e 0: 0
    Similaridade entre Itens 2 e 1: 0
    Similaridade entre Itens 2 e 2: 1
    Similaridade entre Itens 2 e 3: 0.489
    Similaridade entre Itens 2 e 4: 0
    Similaridade entre Itens 2 e 5: 0.447
    Similaridade entre Itens 3 e 0: 0.811
    Similaridade entre Itens 3 e 1: 0.489
    Similaridade entre Itens 3 e 2: 0.489
    Similaridade entre Itens 3 e 3: 1
    Similaridade entre Itens 3 e 4: 0.489
    Similaridade entre Itens 3 e 5: 0.765
    Similaridade entre Itens 4 e 0: 0.772
    Similaridade entre Itens 4 e 1: 1
    Similaridade entre Itens 4 e 2: 0
    Similaridade entre Itens 4 e 3: 0.489
    Similaridade entre Itens 4 e 4: 1
    Similaridade entre Itens 4 e 5: 0
    Similaridade entre Itens 5 e 0: 0.552
    Similaridade entre Itens 5 e 1: 0
    Similaridade entre Itens 5 e 2: 0.447
    Similaridade entre Itens 5 e 3: 0.765
    Similaridade entre Itens 5 e 4: 0
    Similaridade entre Itens 5 e 5: 1


#### 4. Matriz de Similaridades

Podemos imaginar uma outra matriz Itens X Itens, denominada **"Matriz de Similaridades"**, para guardar esses dados de similaridades.


```python
matriz_similaridades = np.empty(NUM_ITENS, dtype=np.ndarray)

for i in range(NUM_ITENS):
    similar = []

    for j in range(NUM_ITENS):
        similar.append(calcular_similaridade(matriz_itens[i], matriz_itens[j]))

    matriz_similaridades[i] = np.array(similar)

matriz_similaridades = np.stack(matriz_similaridades)
```


```python
print(matriz_similaridades)
```

    OUTPUT
    [[1.         0.77151675 0.         0.81059964 0.77151675 0.55205245]
     [0.77151675 1.         0.         0.48867778 1.         0.        ]
     [0.         0.         1.         0.48867778 0.         0.4472136 ]
     [0.81059964 0.48867778 0.48867778 1.         0.48867778 0.76490171]
     [0.77151675 1.         0.         0.48867778 1.         0.        ]
     [0.55205245 0.         0.4472136  0.76490171 0.         1.        ]]


Agora podemos utilizar essa matriz de similaridades para avaliar a utilidade de um item para um usuário.

#### 5. Avaliação de Utilidade

Vamos definir uma função denominada "Avaliar" que pega um usuário e um item, e retorna a Utilidade daquele item para aquele usuário.


```python
def avaliar(usuario: int, item: int) -> float:
    """"
    Dado um usuário e um item, retorna a utilidade daquele item para o usuário.

    Args:
        usuario: int do id do usuário
        item: int do id do item
        k: int do número de usuarios similares usados na comparação

    Returns:
        utilidade: float correspondente a previsão da utilidade/avaliação do 
                   item para o usuário
    """

    # Número de itens similares utilizados no cálculo da utilidade.
    k = 2

    # Primeiro, pegamos o grau de similaridade de cada item com o item 
    # definido no argumento.
    similaridades = matriz_similaridades[item]

    # Agora precisamos obter o id dos k itens mais similares ao item 
    # definido no argumento que foram avaliados pelo usuário definido 
    # no argumento.

    # Ranquear todos itens base no grau de similaridade, e obter o id dos 
    # itens ranqueados.
    itens_ranqueados = list(np.argsort(similaridades)[::-1][1:])

    # Para cada item, se o usuário definido no argumento não o avaliou,
    # remova-o da lista de itens ranqueados.
    for index, item_ranqueado in enumerate(itens_ranqueados):
        if matriz_avaliacoes[usuario, item_ranqueado] == 0:
            itens_ranqueados.pop(index)

    # Obter os k itens mais similares que foram avaliados pelo usuário definido 
    # no argumento.
    itens_mais_similares = itens_ranqueados[:k]

    # Obter as avaliações desses itens do usuário e o grau de similaridade 
    # de cada um.
    avaliacoes_similares_do_usuario = matriz_avaliacoes[usuario, itens_mais_similares]
    grau_similaridade = matriz_similaridades[item, itens_mais_similares]

    # Fazer uma média ponderada das avaliações, onde os pesos são os graus 
    # de similaridade.
    soma_ponderada = 0
    for avaliacao, grau in zip(avaliacoes_similares_do_usuario, grau_similaridade):
        soma_ponderada += grau * avaliacao 
    soma_pesos = sum(grau_similaridade)
    media_ponderada = soma_ponderada / soma_pesos

    # Definimos a utilidade como a média ponderada.
    utilidade = media_ponderada

    return utilidade
```

Agora podemos avaliar a utilidade de cada item para cada usuário:


```python
for u in range(NUM_USUARIOS):
    for i in range(NUM_ITENS):
        print()
        print(f"Avaliando item {i} para usuário {u}: {avaliar(u, i):.3g}", end="")
```

    
    Avaliando item 0 para usuário 0: 4
    Avaliando item 1 para usuário 0: 3.87
    Avaliando item 2 para usuário 0: 4
    Avaliando item 3 para usuário 0: 4.62
    Avaliando item 4 para usuário 0: 3.87
    Avaliando item 5 para usuário 0: 4.42
    Avaliando item 0 para usuário 1: 1.54
    Avaliando item 1 para usuário 1: 1.78
    Avaliando item 2 para usuário 1: 3
    Avaliando item 3 para usuário 1: 0.624
    Avaliando item 4 para usuário 1: 1.78
    Avaliando item 5 para usuário 1: 2.16
    Avaliando item 0 para usuário 2: 0.512
    Avaliando item 1 para usuário 2: 0.388
    Avaliando item 2 para usuário 2: 1
    Avaliando item 3 para usuário 2: 0
    Avaliando item 4 para usuário 2: 0.388
    Avaliando item 5 para usuário 2: 0.631
    Avaliando item 0 para usuário 3: 2.56
    Avaliando item 1 para usuário 3: 4.39
    Avaliando item 2 para usuário 3: 3.57
    Avaliando item 3 para usuário 3: 3.03
    Avaliando item 4 para usuário 3: 4.39
    Avaliando item 5 para usuário 3: 4.58
    Avaliando item 0 para usuário 4: 2.05
    Avaliando item 1 para usuário 4: 1.55
    Avaliando item 2 para usuário 4: 2.57
    Avaliando item 3 para usuário 4: 0.61
    Avaliando item 4 para usuário 4: 1.55
    Avaliando item 5 para usuário 4: 4.37
    Avaliando item 0 para usuário 5: 0
    Avaliando item 1 para usuário 5: 0
    Avaliando item 2 para usuário 5: 0
    Avaliando item 3 para usuário 5: 0
    Avaliando item 4 para usuário 5: 0
    Avaliando item 5 para usuário 5: 0

#### 6. Gerando Recomendações

Vamos criar um ranqueamento dessas avaliações, e recomendar os itens mais bem ranqueados e desconhecidos pelos usuários. Um item desconhecido é um item que o usuário ainda não avaliou.


```python
recomendacoes = []

for u in range(NUM_USUARIOS):
    avaliacoes = []
    for i in range(NUM_ITENS):
        avaliacao = avaliar(u, i)
        if matriz_avaliacoes[u, i] != 0:
            continue
        else:
            avaliacoes.append((i, avaliacao))
    
    avaliacoes_ranqueadas = sorted(avaliacoes, key=(lambda x: x[1]))

    recomendacoes.append((u, avaliacoes_ranqueadas))

print("Recomendações rankeadas\n")
for recomendacao in recomendacoes:
    print(f"Usuário {recomendacao[0]}: ", end='')

    if recomendacao[1]:
        for i in range(2):
            print(f"Item {recomendacao[1][i][0]} ", end='')
    
    print("")
```

    OUTPUT
    Recomendações rankeadas
    
    Usuário 0: Item 2 Item 5 
    Usuário 1: Item 1 Item 4 
    Usuário 2: Item 1 Item 4 
    Usuário 3: Item 2 Item 1 
    Usuário 4: Item 1 Item 4 
    Usuário 5: Item 0 Item 1 

