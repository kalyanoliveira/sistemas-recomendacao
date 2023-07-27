### Implementação básica de Filtragem baseada em Conteúdo

Filtragem baseada em conteúdo é um processo pertencente a sistemas de recomendação, onde o objetivo geral é filtrar uma lista de itens a fim de sugerir aqueles que estimamos ser interessantes para usuários.

Dado informações de características de cada item de um catálogo, um item é avaliado como "relevante" para um usuário se o mesmo demonstrou alguma preferência por uma característica pertencente ao objeto no passado.

Abaixo, uma implementação básica e conceitual de um sistema de recomendação de filmes foi conduzida.

#### 1. Importação de Bibliotecas


```python
import pandas as pd
```

#### 2. Dados de Itens e Usuários

Iremos criar duas `pandas DataFrames` para armazenar dados de itens (filmes) e usuários.

Uma primeira DataFrame conterá *representações* de cada item - neste caso, iremos armazenar o gênero de cada filme.

A segunda DataFrame terá dados de um histórico de *avaliações*. Os dados necessários para construção dele, em um cenário de aplicação real, seriam fornecidos pelos próprios usuários.


```python
representacoes = pd.DataFrame({
    "item": ["Star Wars", "Batman: O Cavaleiro das Trevas", "Shrek", "Os Incríveis", "Avatar", "Missão Impossível", "Pânico", "Hereditário", "O Feiticeiro de Oz", "A Bela e o Monstro"],
    "genero": ["Ficção Científica", "Ação", "Animação", "Animação", "Ficção Científica", "Ação", "Terror", "Terror", "Fantasia", "Fantasia"]
})

avaliacoes = pd.DataFrame({
    "usuario": [0, 0, 1, 2, 2, 3, 3, 3, 4],
    "item": ["Star Wars", "O Feiticeiro de Oz", "A Bela e o Monstro", "Hereditário", "Os Incríveis", "Pânico", "Shrek", "Batman: O Cavaleiro das Trevas", "Avatar"],
    "avaliacao": [5, 3, 4, 2, 5, 1, 4, 4, 3]
})
```

#### 3. Criando Perfis de Usuários

Vamos agora dispor dos dados de avaliações para modelar a preferências de cada usuário no que se diz sobre gêneros de filmes.

Iremos pegar o filme mais bem avaliado de cada usuário, e dizer que o gênero do mesmo é a preferência dele.


```python
for u in range(5):
    avaliacoes_usuario = avaliacoes[avaliacoes["usuario"] == u]

    melhor_avaliacao = avaliacoes_usuario.sort_values(by="avaliacao", ascending=False).values[0]

    genero_preferido = representacoes[representacoes["item"] == melhor_avaliacao[1]]["genero"].values[0]

    print(f"Gênero preferido do usuário {u}: {genero_preferido}")
```

    OUTPUT
    Gênero preferido do usuário 0: Ficção Científica
    Gênero preferido do usuário 1: Fantasia
    Gênero preferido do usuário 2: Animação
    Gênero preferido do usuário 3: Animação
    Gênero preferido do usuário 4: Ficção Científica


Acima, temos os gênero preferido de cada usuário. Eventualmente iremos utiliza-lo para gerar recomendações, mas por enquanto iremos armazenar estes perfis.


```python
perfis = {}

for u in range(5):
    avaliacoes_usuario = avaliacoes[avaliacoes["usuario"] == u]
    melhor_avaliacao = avaliacoes_usuario.sort_values(by="avaliacao", ascending=False).values[0]
    genero_preferido = representacoes[representacoes["item"] == melhor_avaliacao[1]]["genero"].values[0]
    perfis[u] = genero_preferido

print(perfis)
```

    OUTPUT
    {0: 'Ficção Científica', 1: 'Fantasia', 2: 'Animação', 3: 'Animação', 4: 'Ficção Científica'}


#### 4. A Função Avaliar

De uma maneira simplista, uma função "avaliar" para este cenário poderia simplesmente prever a utilidade de um item (filme) para um usuário ao comparar o gênero daquele item com as preferências do usuário em questão.

A utilidade de um filme para um usuário seria um valor binário: sim, ou não.


```python
def avaliar(usuario: int, item: str) -> bool:
    """
    Dado um usuário e um item, retorna a utilidade daquele item para aquele usuário.

    Args:
        usuario: int representando o id do usuário
        item: ttr do nome do item 

    Returns:
        utilidade: bool indicando se o usuário gosta ou não do item
    """

    genero_preferido_usuario = perfis[usuario] 

    filmes_parecidos = representacoes[representacoes["genero"] == genero_preferido_usuario]

    if item not in filmes_parecidos["item"].values:
        utilidade = False
    else:
        utilidade = True

    return utilidade 
```

Sabemos que o usuário `0` avaliou o filme `Star Wars` com uma nota máxima de `5`, portanto inferimos que ele possui uma preferência por filmes do gênero `Ficção Científica`. Isso significa que o filme `Avatar` deve ser útil para ele:


```python
avaliar(0, "Avatar")
```




    OUTPUT
    True



Vamos utilizar a função avaliar para criar um processo de geração de recomendações.


```python
recomendacoes = []

for u in range(5):
    estimativas_usuario = []

    for filme in representacoes["item"].values:
        if (avaliar(u, filme)):
            estimativas_usuario.append((filme))

    recomendacoes.append((u, estimativas_usuario))

for rs in recomendacoes: 
    print(f"Recomendações para usuário {rs[0]}:")
    for r in rs[1]:
        print(r)
```

    OUTPUT
    Recomendações para usuário 0:
    Star Wars
    Avatar
    Recomendações para usuário 1:
    O Feiticeiro de Oz
    A Bela e o Monstro
    Recomendações para usuário 2:
    Shrek
    Os Incríveis
    Recomendações para usuário 3:
    Shrek
    Os Incríveis
    Recomendações para usuário 4:
    Star Wars
    Avatar

