### Implementação Básica de Filtragem baseada em Conteúdo por Semântica Endógena TF-IDF

Filtragem baseada em conteúdo é um processo pertencente a sistemas de recomendação, onde o objetivo geral é filtrar uma lista de itens a fim de sugerir aqueles que estimamos ser interessantes para usuários.

Dado informações de características de cada item de um catálogo, um item é avaliado como "relevante" para um usuário se o mesmo demonstrou alguma preferência por uma característica pertencente ao objeto no passado.

Abaixo, uma implementação de filtragem baseada em conteúdo por semântica endógena TF-IDF de artigos do Wikipedia foi desenvolvida.

#### 1. Importação de Bibliotecas


```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
```

#### 2. Dados de Itens e Usuários

Primeiramente, iremos pegar o primeiro parágrafo de alguns artigos do Wikipedia relacionados a *"Machine Learning"*, e criaremos uma `pandas DataFrame` disso.


```python
titulos = ["Activation function", 
           "AlexNet", 
           "Adaptive website", 
           "AI boom", 
           "Active learning (machine learning)", 
           "Affective computing", 
           "Alan Turing", 
           "Agriculture", 
           "AI winter", 
           "Machine learning", 
           "Action selection", 
           "AI safety", 
           "ADALINE", 
           "AAAI Conference on Artificial Intelligence", 
           "ACM Computing Surveys", 
           "AI takeover", 
           "ACM Computing Classification System", 
           "Adversarial machine learning"]

paragrafos = ["In artificial neural networks, the activation function of a node defines the output of that node given an input or set of inputs.", 
              "AlexNet is the name of a convolutional neural network (CNN) architecture, designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton, who was Krizhevsky's Ph.D. advisor.", 
              "An adaptive website is a website that builds a model of user activity and modifies the information and/or presentation of information to the user in order to better address the user's needs.", 
              "The AI boom (also known as the AI spring) refers to an ongoing period of rapid and unprecedented development in the field of artificial intelligence, with the generative AI race being a key component of this boom, which began in earnest with the founding of OpenAI in 2016 or 2017. OpenAI's generative AI systems, such as its various GPT models (starting in 2018) and DALL-E (2021), have played a significant role in driving this development.", 
              "Active learning is a special case of machine learning in which a learning algorithm can interactively query a user (or some other information source) to label new data points with the desired outputs. In statistics literature, it is sometimes also called optimal experimental design. The information source is also called teacher or oracle.", 
              "Affective computing is the study and development of systems and devices that can recognize, interpret, process, and simulate human affects. It is an interdisciplinary field spanning computer science, psychology, and cognitive science. While some core ideas in the field may be traced as far back as to early philosophical inquiries into emotion, the more modern branch of computer science originated with Rosalind Picard's 1995 paper on affective computing and her book Affective Computing published by MIT Press. One of the motivations for the research is the ability to give machines emotional intelligence, including to simulate empathy. The machine should interpret the emotional state of humans and adapt its behavior to them, giving an appropriate response to those emotions.", 
              "Alan Mathison Turing OBE FRS (/ˈtjʊərɪŋ/; 23 June 1912 – 7 June 1954) was a British mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer. He is widely considered to be the father of theoretical computer science and artificial intelligence.", 
              "Agriculture encompasses crop and livestock production, aquaculture, fisheries and forestry for food and non-food products. Agriculture was the key development in the rise of sedentary human civilization, whereby farming of domesticated species created food surpluses that enabled people to live in cities. While humans started gathering grains at least 105,000 years ago, nascent farmers only began planting them around 11,500 years ago. Sheep, goats, pigs and cattle were domesticated around 10,000 years ago. Plants were independently cultivated in at least 11 regions of the world. In the twentieth century, industrial agriculture based on large-scale monocultures came to dominate agricultural output.", 
              "In the history of artificial intelligence, an AI winter is a period of reduced funding and interest in artificial intelligence research. The term was coined by analogy to the idea of a nuclear winter. The field has experienced several hype cycles, followed by disappointment and criticism, followed by funding cuts, followed by renewed interest years or even decades later.", 
              "Machine learning (ML) is an umbrella term for solving problems for which development of algorithms by human programmers would be cost-prohibitive, and instead the problems are solved by helping machines 'discover' their 'own' algorithms, without needing to be explicitly told what to do by any human-developed algorithms. When there was a vast amount of potential answers, the correct ones needed to be labeled as valid by human labelers initially and human supervision was needed.", 
              "Action selection is a way of characterizing the most basic problem of intelligent systems: what to do next. In artificial intelligence and computational cognitive science, \"the action selection problem\" is typically associated with intelligent agents and animats—artificial systems that exhibit complex behaviour in an agent environment. The term is also sometimes used in ethology or animal behavior.", 
              "AI safety is an interdisciplinary field concerned with preventing accidents, misuse, or other harmful consequences that could result from artificial intelligence (AI) systems. It encompasses machine ethics and AI alignment, which aim to make AI systems moral and beneficial, and AI safety encompasses technical problems including monitoring systems for risks and making them highly reliable. Beyond AI research, it involves developing norms and policies that promote safety.", 
              "ADALINE (Adaptive Linear Neuron or later Adaptive Linear Element) is an early single-layer artificial neural network and the name of the physical device that implemented this network. The network uses memistors. It was developed by professor Bernard Widrow and his doctoral student Ted Hoff at Stanford University in 1960. It is based on the perceptron. It consists of a weight, a bias and a summation function.", 
              "The AAAI Conference on Artificial Intelligence (AAAI) is one of the leading international academic conference in artificial intelligence held annually. Along with ICML, NeurIPS and ICLR, it is one of the primary conferences of high impact in machine learning and artificial intelligence research. It is supported by the Association for the Advancement of Artificial Intelligence. Precise dates vary from year to year, but paper submissions are generally due at the end of August to beginning of September, and the conference is generally held during the following February. The first AAAI was held in 1980 at Stanford University, Stanford California.", 
              "ACM Computing Surveys is a quarterly peer-reviewed scientific journal published by the Association for Computing Machinery. It publishes survey articles and tutorials related to computer science and computing. The journal was established in 1969 with William S. Dorn as founding editor-in-chief.", 
              "An AI takeover is a hypothetical scenario in which artificial intelligence (AI) becomes the dominant form of intelligence on Earth, as computer programs or robots effectively take control of the planet away from the human species. Possible scenarios include replacement of the entire human workforce, takeover by a superintelligent AI, and the popular notion of a robot uprising. Stories of AI takeovers are very popular throughout science fiction. Some public figures, such as Stephen Hawking and Elon Musk, have advocated research into precautionary measures to ensure future superintelligent machines remain under human control.", 
              "The ACM Computing Classification System (CCS) is a subject classification system for computing devised by the Association for Computing Machinery (ACM). The system is comparable to the Mathematics Subject Classification (MSC) in scope, aims, and structure, being used by the various ACM journals to organize subjects by area.", 
              "Adversarial machine learning is the study of the attacks on machine learning algorithms, and of the defenses against such attacks. A survey from May 2020 exposes the fact that practitioners report a dire need for better protecting machine learning systems in industrial applications."]

conteudos = pd.DataFrame({"item": titulos,
                          "conteudo": paragrafos})
```

Agora, iremos criar outra `pandas DataFrame` com feedback explícito (avaliações) de 5 usuários.


```python
avaliacoes = pd.DataFrame({
    "usuario":[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    "item":["Activation function", "Agriculture", "Alan Turing", "AI takeover", "ACM Computing Surveys", "ADALINE", "AAAI Conference on Artificial Intelligence", "Adversarial machine learning", "Affective computing", "AlexNet", "AI winter", "AI safety", "ACM Computing Classification System", "Adaptive website", "AI boom"],
    "avaliacao":[5, 4, 3, 5, 3, 5, 5, 3, 4, 5, 1, 3, 4, 1, 1]
})
```

#### 3. Representações de Itens

Iremos agora criar representações de cada artigo utilizando uma técnica chamada TF-IDF.

Cada documento será representado por um vetor. Cada dimensão desse vetor representará a importância de uma palavra (ou combinação de palavras) para aquele documento. O número de dimensões é igual o número de palavras diferentes na combinação de todos os documentos.

Utilizamos TF-IDF, que significa *"Term Frequency-Inverse Document Frequency"*. Para calcular a relevância de um termo - uma palavra ou combinação de palavras - para um documento fictício "A", contamos a quantidade de vezes que este termo aparece no documento A, e multiplicamos este valor pelo inverso da quantidade de vezes que este termo aparece em outros documentos que não o "A".

A consequência disto é que termos que aparecem muitas vezes em um documento e poucas em outros são considerados mais importantes para aquele documento.

A biblioteca `sklearn` possui um objeto `TfidfVectorizer` que cria uma matriz TF-IDF. Uma matriz TF-IDF possui termos nas linhas e "documentos" nas colunas, de modo que cada valor da matriz representa a relevância de um termo para um documento.

No nosso caso, estamos considerando que um "termo" é composto de uma ou duas palavras (`ngram_range(1, 2)`), e nossos "documentos" são os primeiros parágrafos de cada artigo selecionado do Wikipedia.


```python
vetorizador_tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=0, stop_words="english")
```

Para criar a matriz "termos-documentos", podemos utilizar o método `fit_transform` do objeto `vetorizador_tfidf`, passando a coluna `"conteudo"` da nossa `DataFrame` `conteudos`.


```python
matriz_tfidf = vetorizador_tfidf.fit_transform(conteudos["conteudo"])

print(matriz_tfidf.toarray())
print(matriz_tfidf.shape)
```

    OUTPUT
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    (18, 1083)


Como podemos ver acima, nossa matriz possui os `18` documentos que selecionamos acima, e também possui `1083` termos diferentes.

A maioria dos valores desta matriz serão `0`. Podemos visualizar os `top 10` termos mais relevantes de cada documento com o código abaixo:


```python
df = pd.DataFrame(data = matriz_tfidf.toarray(), index=titulos, columns=vetorizador_tfidf.get_feature_names_out())
n = 10
for doc in range(len(conteudos["item"])):
    print(titulos[doc])
    sorted_indices = np.argsort(df.values[doc])[::-1]
    for i in sorted_indices[:n]:
        print(df.values[doc][i], vetorizador_tfidf.get_feature_names_out()[i])
    print()
```

    OUTPUT
    Activation function
    0.39862724922926296 node
    0.19931362461463148 node given
    0.19931362461463148 networks
    0.19931362461463148 given input
    0.19931362461463148 output node
    0.19931362461463148 function node
    0.19931362461463148 activation function
    0.19931362461463148 activation
    0.19931362461463148 node defines
    0.19931362461463148 defines output
    
    AlexNet
    0.3422363546964575 krizhevsky
    0.17111817734822876 advisor
    0.17111817734822876 ph
    0.17111817734822876 ph advisor
    0.17111817734822876 cnn
    0.17111817734822876 cnn architecture
    0.17111817734822876 designed
    0.17111817734822876 designed alex
    0.17111817734822876 collaboration
    0.17111817734822876 alexnet convolutional
    
    Adaptive website
    0.41909147963799404 user
    0.3192016096558265 website
    0.2793943197586627 information
    0.15960080482791325 needs
    0.15960080482791325 address
    0.15960080482791325 website website
    0.15960080482791325 website builds
    0.15960080482791325 better address
    0.15960080482791325 adaptive website
    0.15960080482791325 builds
    
    AI boom
    0.30208940228783404 ai
    0.2103169908780517 openai
    0.2103169908780517 generative ai
    0.2103169908780517 generative
    0.2103169908780517 boom
    0.1392508285627045 development
    0.10515849543902585 boom began
    0.10515849543902585 rapid
    0.10515849543902585 race key
    0.10515849543902585 race
    
    Active learning (machine learning)
    0.26503637693084897 learning
    0.24602718200711543 called
    0.24602718200711543 information source
    0.24602718200711543 source
    0.21534539638799094 information
    0.12301359100355772 new data
    0.12301359100355772 new
    0.12301359100355772 active learning
    0.12301359100355772 active
    0.12301359100355772 algorithm interactively
    
    Affective computing
    0.24776068936819612 affective
    0.24776068936819612 affective computing
    0.1949402616961062 computing
    0.16517379291213075 simulate
    0.16517379291213075 emotional
    0.16517379291213075 interpret
    0.1640422922358795 science
    0.1299601744640708 computer science
    0.11862392136299635 computer
    0.11862392136299635 field
    
    Alan Turing
    0.2842091072681879 turing
    0.2842091072681879 theoretical
    0.2721497012072539 computer
    0.1894727381787919 theoretical computer
    0.1894727381787919 considered
    0.1894727381787919 june
    0.14907879558714526 computer science
    0.1254498539147488 science
    0.09473636908939595 ˈtjʊərɪŋ 23
    0.09473636908939595 algorithm computation
    
    Agriculture
    0.22614736081025125 years ago
    0.22614736081025125 agriculture
    0.22614736081025125 ago
    0.22614736081025125 food
    0.19794476634038388 years
    0.15076490720683416 000
    0.15076490720683416 000 years
    0.15076490720683416 domesticated
    0.15076490720683416 11
    0.07538245360341708 farming domesticated
    
    AI winter
    0.36578852693503655 followed
    0.24385901795669102 funding
    0.24385901795669102 winter
    0.13988176754610415 artificial intelligence
    0.13104760033311083 intelligence
    0.12314516847795093 artificial
    0.12192950897834551 hype cycles
    0.12192950897834551 hype
    0.12192950897834551 nuclear
    0.12192950897834551 followed renewed
    
    Machine learning
    0.30795907510176196 human
    0.2814982082852626 algorithms
    0.2144035027328694 needed
    0.18766547219017507 problems
    0.1072017513664347 labelers
    0.1072017513664347 supervision needed
    0.1072017513664347 ones
    0.1072017513664347 ones needed
    0.1072017513664347 learning ml
    0.1072017513664347 discover algorithms
    
    Action selection
    0.23817398124886516 action selection
    0.23817398124886516 problem
    0.23817398124886516 action
    0.23817398124886516 selection
    0.23817398124886516 intelligent
    0.15769493511921318 systems
    0.12027430969628845 artificial
    0.11908699062443258 intelligent agents
    0.11908699062443258 intelligent systems
    0.11908699062443258 animal
    
    AI safety
    0.41226381440400617 ai
    0.287021273294267 safety
    0.19134751552951132 ai safety
    0.1900367153147592 systems
    0.16748477238733783 ai systems
    0.16748477238733783 encompasses
    0.09567375776475566 accidents misuse
    0.09567375776475566 misuse
    0.09567375776475566 harmful
    0.09567375776475566 harmful consequences
    
    ADALINE
    0.28910758911492135 network
    0.2201991982488206 linear
    0.2201991982488206 adaptive linear
    0.19273839274328092 adaptive
    0.1100995991244103 developed professor
    0.1100995991244103 bernard
    0.1100995991244103 physical
    0.1100995991244103 bernard widrow
    0.1100995991244103 perceptron consists
    0.1100995991244103 perceptron
    
    AAAI Conference on Artificial Intelligence
    0.2626003064974002 conference
    0.2626003064974002 held
    0.2626003064974002 aaai
    0.20084279481804937 artificial intelligence
    0.18815866261073533 intelligence
    0.17681231971349975 artificial
    0.1750668709982668 conference artificial
    0.1750668709982668 year
    0.1750668709982668 generally
    0.15323446954912776 stanford
    
    ACM Computing Surveys
    0.31572704239599486 computing
    0.26751699553359753 journal
    0.13375849776679877 reviewed
    0.13375849776679877 reviewed scientific
    0.13375849776679877 scientific journal
    0.13375849776679877 related computer
    0.13375849776679877 related
    0.13375849776679877 published association
    0.13375849776679877 science computing
    0.13375849776679877 publishes
    
    AI takeover
    0.2600501115885008 ai
    0.19503758369137558 human
    0.18104890980148655 popular
    0.18104890980148655 control
    0.18104890980148655 superintelligent
    0.18104890980148655 takeover
    0.09729402410955475 intelligence
    0.09052445490074328 form
    0.09052445490074328 stephen
    0.09052445490074328 elon
    
    ACM Computing Classification System
    0.36465521588198513 classification
    0.31917945557244726 acm
    0.2869138901500379 computing
    0.24310347725465675 subject
    0.24310347725465675 subject classification
    0.12155173862732838 journals
    0.12155173862732838 devised association
    0.12155173862732838 mathematics
    0.12155173862732838 comparable
    0.12155173862732838 structure
    
    Adversarial machine learning
    0.29731258873660277 learning
    0.29731258873660277 machine learning
    0.2759884481864605 attacks
    0.23746765002648867 machine
    0.13799422409323026 2020 exposes
    0.13799422409323026 defenses
    0.13799422409323026 defenses attacks
    0.13799422409323026 algorithms defenses
    0.13799422409323026 study attacks
    0.13799422409323026 better protecting
    


#### 4. Gerando Perfis de Usuários

Vamos agora utilizar as representações criadas e os dados de interações de usuários para gerar um perfil para cada usuário do nosso sistema.

Para cada usuário, pegaremos os itens mais bem avaliados por ele, e calcularemos a média das representações deles. Como cada item é representado por um vetor TF-IDF, é a média de vetores que estaremos calculando e chamando de "perfil".


```python
def gerar_perfis(representacoes, avaliacoes, n=2):
    """
    Gera perfis de usuários a partir da disponibilidade de dados de avaliações e representações de artigos Wikipedia em TF-IDF.

    Params:
        representacoes: numpy ndarray, a matriz TF-IDF.
        avaliacoes: pandas DataFrame de avaliações do usuários.

    Returns:
        perfis: list de numpy ndarrays com shape de ndarrays de representações, cada um representando a preferência de usuários por características em itens.
    """

    perfis = []

    # Para cada usuário:
    for u in range(5):

        # Recolher o nome dos itens preferidos do usuário (aqueles mais bem avaliados).
        itens_preferidos_usuario = avaliacoes[avaliacoes["usuario"] == u].sort_values(by="avaliacao")[::-1][:n]["item"].values

        # Recolher o índice dos itens preferidos do usuário
        indices_itens_preferidos_usuario = []
        for index, t in enumerate(titulos):
            if t in itens_preferidos_usuario:
                indices_itens_preferidos_usuario.append(index)

        # Recolher as representações dos itens preferidos do usuário, que são vetores.
        representacoes_itens_preferidos_usuario = []
        for index in indices_itens_preferidos_usuario:
            representacoes_itens_preferidos_usuario.append(representacoes.toarray()[index])

        # Fazer a média das representações/vetores.
        soma = np.zeros_like(representacoes_itens_preferidos_usuario[0])
        count = 0
        for r in representacoes_itens_preferidos_usuario:
            soma += r
            count += 1
        media = soma / count

        # A média calculada acima é um vetor que representa a preferência do usuário por cada termo presente no corpus.
        # Isto é, a média é um vetor que pode ser comparado com outros documentos para buscar uma similaridade.
        # Vamos salvar esse vetor na lista de perfis para uso futuro.
        perfis.append(media)

    return perfis
```

Com esta função, conseguimos gerar nossos perfis de usuários, dispondo das representações de itens `matriz_tfidf` e interações de usuários `avaliacoes`.


```python
perfis = np.stack(np.array(gerar_perfis(matriz_tfidf, avaliacoes)))
```

Com esses componentes, já conseguimos comparar perfis de usuários e representações de itens.

#### 5. Comparando Perfis e Representações

Cada perfil consiste de um vetor com dimensionalidade igual aos vetores de representações de itens. Podemos utilizar a função de similaridade de vetores introduzida anteriormente:


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

Por exemplo, a similaridade entre o perfil do usuário `0` e a representação do artigo `Activation function` pode ser calculada assim:


```python
perfil = perfis[0]
representacao = matriz_tfidf.toarray()[0]

similaridade = calcular_similaridade(perfil, representacao)
print(similaridade)
```

    OUTPUT
    0.7255362555579096


#### 6. Componente de Filtragem, Gerando Recomendações.

Vamos criar um componente de filtragem que pega o perfil de um usuário e retorna uma recomendação dispondo desta função de cálculo de similaridade e, claro, das nossas representações geradas.


```python
def filtrar(perfil, representacoes):
    """
    Dado um perfil de um usuário e representações de artigos Wikipedia em TF-IDF, retorna uma recomendação de um artigo baseado no perfil providenciado. 

    Params:
        perfil: vetor do perfil do usuário.
        representacoes: matriz tf-idf.
    
    Returns:
        recomendacao: nome de um artigo recomendado para o usuário correspondente ao perfil.
    """
    
    # Calcular a similaridade de cada representação com o perfil providenciado.
    similaridades = []
    for r in representacoes:
        similaridades.append(calcular_similaridade(perfil, r))
    
    # Pegar os índices dos artigos mais similares ao perfil do usuário.
    sorted_indices = np.argsort(similaridades)[::-1]
    
    # Estaremos usando [2:3], pois os dois artigos mais favoritos de usuários foram utilizados durante a geração de perfis.
    recomendacao = np.array(titulos)[sorted_indices[2:3]][0]

    return recomendacao
```

Podemos simplesmente chamar essa função para cada perfil de usuário.


```python
for u in range(5):
    print(f"Recomendação de leitura para o usuário {u}: ", end="")
    print(filtrar(perfis[u], matriz_tfidf.toarray()))
```

    OUTPUT
    Recomendação de leitura para o usuário 0: AlexNet
    Recomendação de leitura para o usuário 1: AI winter
    Recomendação de leitura para o usuário 2: ACM Computing Surveys
    Recomendação de leitura para o usuário 3: AI safety
    Recomendação de leitura para o usuário 4: ACM Computing Classification System

