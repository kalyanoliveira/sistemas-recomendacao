### Implementação Básica de Filtragem Colaborativa baseada em Usuários

Filtragem colaborativa é um processo oriundo de sistemas de recomendação, onde o objetivo é filtrar uma lista de itens para sugerir a um usuário o item mais "relevante" dessa lista.

Itens são avaliados como "relevante" a partir de uma análise que busca por similaridades em um histórico de avaliações.

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

Vamos transformar esses dados em uma matrix (um array 2D).


```python
avaliacoes = [usuario0, usuario1, usuario2, usuario3, usuario4, usuario5]

matrix_avaliacoes = np.array(avaliacoes)
```


```python
print(matrix_avaliacoes)
```

    [[5 3 0 4 4 0]
     [1 0 0 3 0 0]
     [0 0 0 1 0 0]
     [4 0 0 5 0 2]
     [0 0 5 4 0 1]
     [0 0 0 0 0 0]]


Cada linha dessa matrix contém um vetor de todas avaliações de um usuário.

Podemos comparar esse vetores para ver o quão similar dois usuários são.

#### 3. Calculando a similaridade entre dois usuários.

Vamos definir uma função que pega dois usuários e retorna a similaridade entre eles.


```python
def calcular_similaridade(vetor1: np.ndarray, vetor2: np.ndarray) -> float:
    """
    Calcula a similaridade entre dois vetores a partir do coseno do ângulo entre
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

Com esses materiais, já conseguimos calcular a similaridade entre usuarios.


```python
for i, usuario in enumerate(matrix_avaliacoes):
    for j, outro_usuario in enumerate(matrix_avaliacoes):
        sim = calcular_similaridade(usuario, outro_usuario)
        print(f"Similaridade entre Usuários {i} e {j}: {sim:.3g}")
```

    Similaridade entre Usuários 0 e 0: 1
    Similaridade entre Usuários 0 e 1: 0.662
    Similaridade entre Usuários 0 e 2: 0.492
    Similaridade entre Usuários 0 e 3: 0.734
    Similaridade entre Usuários 0 e 4: 0.304
    Similaridade entre Usuários 0 e 5: nan
    Similaridade entre Usuários 1 e 0: 0.662
    Similaridade entre Usuários 1 e 1: 1
    Similaridade entre Usuários 1 e 2: 0.949
    Similaridade entre Usuários 1 e 3: 0.896
    Similaridade entre Usuários 1 e 4: 0.586
    Similaridade entre Usuários 1 e 5: nan
    Similaridade entre Usuários 2 e 0: 0.492
    Similaridade entre Usuários 2 e 1: 0.949
    Similaridade entre Usuários 2 e 2: 1
    Similaridade entre Usuários 2 e 3: 0.745
    Similaridade entre Usuários 2 e 4: 0.617
    Similaridade entre Usuários 2 e 5: nan
    Similaridade entre Usuários 3 e 0: 0.734
    Similaridade entre Usuários 3 e 1: 0.896
    Similaridade entre Usuários 3 e 2: 0.745
    Similaridade entre Usuários 3 e 3: 1
    Similaridade entre Usuários 3 e 4: 0.506
    Similaridade entre Usuários 3 e 5: nan
    Similaridade entre Usuários 4 e 0: 0.304
    Similaridade entre Usuários 4 e 1: 0.586
    Similaridade entre Usuários 4 e 2: 0.617
    Similaridade entre Usuários 4 e 3: 0.506
    Similaridade entre Usuários 4 e 4: 1
    Similaridade entre Usuários 4 e 5: nan
    Similaridade entre Usuários 5 e 0: nan
    Similaridade entre Usuários 5 e 1: nan
    Similaridade entre Usuários 5 e 2: nan
    Similaridade entre Usuários 5 e 3: nan
    Similaridade entre Usuários 5 e 4: nan
    Similaridade entre Usuários 5 e 5: nan


    /tmp/ipykernel_1493/2408305389.py:19: RuntimeWarning: invalid value encountered in true_divide
      similaridade = dot / (magnitude1 * magnitude2)


#### 4. Matrix de Similaridades

Podemos imaginar uma outra matrix Usuários X Usuários, denominada **"Matrix de Similaridades"**, para guardar esses dados de similaridades.


```python
matrix_similaridades = np.empty(NUM_USUARIOS, dtype=np.ndarray)

for i in range(NUM_USUARIOS):
    similar = []

    for j in range(NUM_USUARIOS):
        similar.append(calcular_similaridade(matrix_avaliacoes[i], matrix_avaliacoes[j]))

    matrix_similaridades[i] = np.array(similar)

matrix_similaridades = np.stack(matrix_similaridades)
```

    /tmp/ipykernel_1493/2408305389.py:19: RuntimeWarning: invalid value encountered in true_divide
      similaridade = dot / (magnitude1 * magnitude2)



```python
print(matrix_similaridades)
```

    [[1.         0.6617241  0.49236596 0.73397584 0.30389487        nan]
     [0.6617241  1.         0.9486833  0.89566859 0.58554004        nan]
     [0.49236596 0.9486833  1.         0.74535599 0.6172134         nan]
     [0.73397584 0.89566859 0.74535599 1.         0.50604808        nan]
     [0.30389487 0.58554004 0.6172134  0.50604808 1.                nan]
     [       nan        nan        nan        nan        nan        nan]]


Outra maneira mais rápida de calcular essa matrix é a partir dessa linha de código:


```python
matrix_similaridades = np.dot(matrix_avaliacoes, matrix_avaliacoes.T) / (np.linalg.norm(matrix_avaliacoes, axis=1)[:, None] * np.linalg.norm(matrix_avaliacoes.T, axis=0))
```

    /tmp/ipykernel_1493/786230641.py:1: RuntimeWarning: invalid value encountered in true_divide
      matrix_similaridades = np.dot(matrix_avaliacoes, matrix_avaliacoes.T) / (np.linalg.norm(matrix_avaliacoes, axis=1)[:, None] * np.linalg.norm(matrix_avaliacoes.T, axis=0))



```python
print(matrix_similaridades)
```

    [[1.         0.6617241  0.49236596 0.73397584 0.30389487        nan]
     [0.6617241  1.         0.9486833  0.89566859 0.58554004        nan]
     [0.49236596 0.9486833  1.         0.74535599 0.6172134         nan]
     [0.73397584 0.89566859 0.74535599 1.         0.50604808        nan]
     [0.30389487 0.58554004 0.6172134  0.50604808 1.                nan]
     [       nan        nan        nan        nan        nan        nan]]


Agora podemos utilizar essa matrix de similaridades para avaliar a utilidade de um item para um usuário.

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

    # Número de usuários similares utilizados no cálculo da utilidade.
    k = 2

    # Primeiro, pegamos o grau de similaridade de cada usuário com o usuário 
    # definido no argumento.
    similaridades = matrix_similaridades[usuario]

    # Agora precisamos obter o id dos k usuários mais similares ao usuário 
    # definido no argumento que avaliaram o item definido no argumento.

    # Rankear todos usuários com base no grau de similaridade, e obter o id dos 
    # usuários rankeados.
    usuarios_rankeados = list(np.argsort(similaridades)[::-1][1:])

    # Para cada usuário, se ele não avaliou o item definido no argumento,
    # remova-o da lista de usuários similares.
    for index, usuario_rankeado in enumerate(usuarios_rankeados):
        if matrix_avaliacoes[usuario_rankeado, item] == 0:
            usuarios_rankeados.pop(index)

    # Obter os k usuários mais similares que avaliaram o item definido 
    # no argumento.
    usuarios_mais_similares = usuarios_rankeados[:k]

    # Obter as avaliações desses usuários e o grau de similaridade de cada um.
    avaliacoes_similares_do_item = matrix_avaliacoes[usuarios_mais_similares, item]
    grau_similaridade = matrix_similaridades[usuario, usuarios_mais_similares]

    # Fazer uma média ponderada das avaliações, onde os pesos são os graus 
    # de similaridade.
    soma_ponderada = 0
    for avaliacao, grau in zip(avaliacoes_similares_do_item, grau_similaridade):
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

    
    Avaliando item 0 para usuário 0: 4.58
    Avaliando item 1 para usuário 0: 1.81
    Avaliando item 2 para usuário 0: 0
    Avaliando item 3 para usuário 0: 4.42
    Avaliando item 4 para usuário 0: 2.41
    Avaliando item 5 para usuário 0: 1.2
    Avaliando item 0 para usuário 1: 2.42
    Avaliando item 1 para usuário 1: 1.23
    Avaliando item 2 para usuário 1: 0
    Avaliando item 3 para usuário 1: 2.03
    Avaliando item 4 para usuário 1: 1.64
    Avaliando item 5 para usuário 1: 0.971
    Avaliando item 0 para usuário 2: 2.32
    Avaliando item 1 para usuário 2: 0
    Avaliando item 2 para usuário 2: 1.97
    Avaliando item 3 para usuário 2: 1.97
    Avaliando item 4 para usuário 2: 0
    Avaliando item 5 para usuário 2: 0.88
    Avaliando item 0 para usuário 3: 2.58
    Avaliando item 1 para usuário 3: 1.35
    Avaliando item 2 para usuário 3: 0
    Avaliando item 3 para usuário 3: 4.06
    Avaliando item 4 para usuário 3: 1.8
    Avaliando item 5 para usuário 3: 1.15
    Avaliando item 0 para usuário 4: 0.487
    Avaliando item 1 para usuário 4: 0
    Avaliando item 2 para usuário 4: 3.15
    Avaliando item 3 para usuário 4: 2.86
    Avaliando item 4 para usuário 4: 0
    Avaliando item 5 para usuário 4: 0.631
    Avaliando item 0 para usuário 5: nan
    Avaliando item 1 para usuário 5: nan
    Avaliando item 2 para usuário 5: nan
    Avaliando item 3 para usuário 5: nan
    Avaliando item 4 para usuário 5: nan
    Avaliando item 5 para usuário 5: nan

#### 6. Gerando Recomendações

Vamos criar um ranqueamento dessas avaliações, e recomendar os itens mais bem ranqueados e desconhecidos pelos usuários. Um item desconhecido é um item que o usuário ainda não avaliou.


```python
recomendacoes = []

for u in range(NUM_USUARIOS):
    avaliacoes = []
    for i in range(NUM_ITENS):
        avaliacao = avaliar(u, i)
        if matrix_avaliacoes[u, i] != 0:
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

    Recomendações rankeadas
    
    Usuário 0: Item 2 Item 5 
    Usuário 1: Item 2 Item 5 
    Usuário 2: Item 1 Item 4 
    Usuário 3: Item 2 Item 1 
    Usuário 4: Item 1 Item 4 
    Usuário 5: Item 0 Item 1 

