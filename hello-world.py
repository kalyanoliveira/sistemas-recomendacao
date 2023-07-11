"""
Implementação básica de um sistema de recomendação não personalizado a partir da
disponibilidade de dados de avaliações.

Todos sistemas de recomendação utilizam dados categorizados como interações.
Interações se configuram toda vez que um usuário do sistema interage com um 
item do catálogo. 

Feedback explícito, ou avaliações, são exemplos de interações que conseguimos
utilizar em sistemas de recomendação.
"""

# Importação de bibliotecas.
import numpy as np

# Usaremos dados de um histórico de avaliações para os processos subsequentes.
# Para cada usuário, um array contém as avaliações daquele usuário para cada 
# item de um catálogo:
# usuario = [item0, item1, item2, item3, ...]
# Caso um usuário não tenha avaliado um item, deixamos a avaliação como 0.
usuario0 = [5, 1, 0, 2, 2]
usuario1 = [1, 5, 2, 5, 5]
usuario2 = [2, 0, 3, 5, 4]
usuario3 = [4, 3, 5, 3, 0]

# Vamos transformar esses dados em uma matrix (um array 2D).
# Cada linha dessa matrix contém um vetor de todas avaliações de um usuário.
matrix_avaliacoes = np.array([usuario0, usuario1, usuario2, usuario3])

# print(matrix_avaliacoes)
# [[5 1 0 2 2]
#  [1 5 2 5 5]
#  [2 0 3 5 4]
#  [4 3 5 3 0]]

# Agora podemos recomendar aquele item com a melhor avaliação média.
avaliacoes_medias = np.mean(matrix_avaliacoes, axis=0)
melhor_item = np.argsort(avaliacoes_medias)[::-1][0]
print(f"Recomendação de item (aquele mais bem avaliado): Item {melhor_item}")