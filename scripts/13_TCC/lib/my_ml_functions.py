
# Bibliotecas Básicas
import os
import sys
import string
from unicodedata import normalize
import time
import datetime as dt
from numpy.lib.function_base import average
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Processamento dos Dados
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample
from sklearn.decomposition import TruncatedSVD

# Exploração do dados
from sklearn import metrics
from sklearn.datasets import make_classification
from nltk import ngrams, FreqDist
from matplotlib_venn import venn2 


# Modelagem / Machine Learning
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, \
                                    cross_val_score, cross_val_predict, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



#==============================================================================================================
######################################## ANÁLISE EXPLORATÓRIA DOS DADOS########################################
#==============================================================================================================

# Cria uma tabela no formato latex 
def make_latex(df, file_name, index=False, max_colwidth=True):
    if max_colwidth == True:
        with pd.option_context("max_colwidth", None): # impede de trucar o texto em linhas muito longas
            df.to_latex(f'out/{file_name}.tex', index=index)
    else:
        df.to_latex(f'out/{file_name}.tex', index=index)

# inserida
def plot_target_count(df, target, figsize=(9,4), savefig=False, title=True):
    """Plota gráfico com a contagem de registros em cada classificação

    Args:
        df (Pandas Dataframe): nome do dataframe onde estão os dados a serem plotados
        target (pd.DataFrame column): nome da coluna onde está os labels
        fig_size (tuple, optional): tamanho da imagem. Defaults to (9,4).
    """
    # Quantidade de Registros no DataFrame
    ncount = len(df)    

    ax = plt.subplots(figsize=figsize)
    ax = sns.countplot(x=df[target])
    plt.xticks(fontsize=11)
    if title == True:
        plt.title('Distruição das Classes de Avaliação',fontsize=14, pad=10)
    ax.set_ylabel('')
    ax.set_xlabel('Score', size=10)
    # Insererótulo de percentual
    for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.1f}%'.format(100 * y / ncount), (x.mean(), y), ha='center', va='bottom')
    # Salve como arquivo png
    if savefig == True:
        plt.savefig(f'out/target_count_{dt.date.today()}.png', dpi=300, pad_inches=0, bbox_inches = 'tight')
        
        
# função para plotar os termos mais frequentes
# por padrão, irá plotar os top 30
def freq_words_df_or_plot(x, terms=30, plot=True, savefig=False):

    all_words = ' '.join([text for text in x]) 
    all_words = all_words.split() 

    fdist = FreqDist(all_words) 
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    words_df = words_df.nlargest(columns="count", n = terms) 
    if plot == True:
        # selecinando as n_terms palavras mais frequentes     
        plt.figure(figsize=(16,8))
        plt.title(f'Os {terms} termos mais frequentes')
        ax = sns.barplot(data=words_df, x= "count", y = "word") 
        ax.set(xlabel='contagem')
        ax.set(ylabel='termos') 
        # Salve como arquivo png
        if savefig == True:
            plt.savefig(f'out/freq_words_{dt.date.today()}.png', dpi=300, pad_inches=0, bbox_inches = 'tight')
        plt.show()
    else:
        return words_df

#inserido
def plot_from_dict_freq_words(words_dict, n_rows=1, n_cols=2, savefig=False):
    """Plota a frequência de palavras a partir de um dicionários de dataframes

    Args:
        words_dict (python dicionário): dicionário com um nome do dataframe e o datafram
        n_rows (int, optional): número de linhas do plot. Defaults to 1.
        n_cols (int, optional): número de colunas do plot. Defaults to 2.
        savefig (bool, optional): salvar o plot em disco. Defaults to False.
    """
    # selecinando as n_terms palavras mais frequentes
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5))
    i = 0
    colors = ['Blues_d', 'Reds_d']
    for title, data in words_dict.items():
        ax = axs[i]
        sns.barplot(data=data, x= "count", y = "word", ax=ax, palette=colors[i])
        ax.set_title(title, size=12)
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Incrementa o index
        i +=1
    plt.tight_layout()
    if savefig == True:
        plt.savefig(f'out/freq_words_from_dict_{dt.date.today()}.png', dpi=300, pad_inches=0, bbox_inches = 'tight')
    plt.show()

# Função para contar quantas vezes cada ngram aparece
def word_vec_ngrams_counter(data, n_grams):
    # Inicializa a função
    cv = CountVectorizer(ngram_range=n_grams).fit(data)
    # Faz a tokenização e constroi o vocabulário
    bag_of_words = cv.transform(data)
    # Conta quantas vezes cada n_grams aparece
    sum_bow = bag_of_words.sum(axis=0)
    # Faz a associação do n_grams com a contagem
    words_freq = [(word, sum_bow[0, idx]) for word, idx in cv.vocabulary_.items()]
    
    # Retorna um DataFrame com a contagem dos n_grams
    count_df = pd.DataFrame(words_freq, columns=['word', 'count'])
    return count_df

# Função para plotar diagram de Venn com o conjunto de palavras segregado por polaridade
def plot_venn(pos_list, neg_list, title, fig_size=(11,6), savefig=False):
    # Conjunto com a interseção das palavras positivas e negativas
    common_set = len(set(neg_list).intersection(pos_list))
    # Conjunto com as palvaras positivas que não estão no cojunto das negativas
    pos_set = len(set(pos_list).difference(neg_list))
    # Conjunto das palvras negativas que não estão no conjunto das palvras positivas
    neg_set = len(set(neg_list).difference(pos_list))
    total = pos_set + neg_set + common_set

    # Plota um Diagaram de Venn
    plt.figure(figsize=fig_size)
    plt.title(title, size=14)
    venn2(
        subsets=(pos_set, neg_set, common_set),
        set_labels=('Positive', 'Negative'),
        subset_label_formatter=lambda x: f"{(x/total):1.0%}",
        set_colors=['blue', 'red'],
        alpha=0.5
    )
    if savefig == True:
        plt.savefig(f'out/venn_{dt.date.today()}.png', dpi=300, pad_inches=0, bbox_inches = 'tight')

# Análise do efeito do desbalanceamento dos dados
# label latex: cod:FuncImbalancedAnalysis
def imbalanced_analysis(data_dict, savefig=False):
    # bibliotecas necessárias
    plt.figure(figsize=(10, 5))
    d = []
    k = 1


    for df_name, df_data in data_dict.items():
        
        corpus = df_data['processed_review']
        y = df_data['score']

        X_train, X_test, y_train, y_test = train_test_split(
            corpus, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # Instanciação do Modelo
        cv = CountVectorizer(min_df=2)
        cv.fit(X_train)
        X_train = cv.transform(X_train)
        X_test = cv.transform(X_test)

        classifier_lgr = LogisticRegression()
        classifier_lgr.fit(X_train, y_train)
        y_pred = classifier_lgr.predict(X_test)

        # Chama a funçao da Matriz da Confusão e Plota
        plt.subplot(1, 2, k)
        custom_cf_matrix(true_labels=y_test, predicted_labels=y_pred, data_name=df_name)
        k += 1

        # Performance do Modelo
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)

        # Dicinoário com cada métrica do modelo
        test_performance = {}
        test_performance['dataset'] = df_name
        test_performance['acurácia'] = round(accuracy, 4)
        test_performance['precisão'] = round(precision, 4)
        test_performance['revocação'] = round(recall, 4)
        test_performance['f1_score'] = round(f1, 4)
        
        # Salva em uma lista o resultado de avaliação do modelo
        d.append(test_performance)
    
    # Cria o DataFrame com a formatação em percentual
    
    df = pd.DataFrame(d)
    for column in ['acurácia', 'precisão', 'revocação', 'f1_score']:
        df[column] = df[column].map('{:.2%}'.format)
    
    plt.tight_layout()
    if savefig == True:
        plt.savefig(f'out/imbalanced_{dt.date.today()}.png', dpi=300, pad_inches=0, bbox_inches = 'tight')
    plt.show()

    print('Relatório de Classificação do modelo com o Dataset Balanceado vs Desbalanceado:')
    print(df)
    return df

def vectorize(df):
    """Extraí as features do corpus usando o CountVectorizer e faz a
    redução da dimensionalidade

    Args:
        df (pandas dataframe): dataframe com duas colunas, sendo a primeira o corpus a
        segunda os rótulos

    Returns:
        pandas dataframe: retorna uma pandas datafram com 1 dimensão
    """
    corpus = df.iloc[:,0]

    y = df.iloc[:,-1]
    # Cria o saco de palavras
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)

    # Reduz o corpus para 1 dimensão
    pca = TruncatedSVD(n_components=1)
    pca_ = pca.fit_transform(X)
    # Salva o dataset com 1 dimensão em uma Pandas DataFrame
    df_pca = pd.DataFrame(pca_, columns=['features'])
    df_pca['target'] = y

    return df_pca


#==============================================================================================================
############################################ PROCESSAMENTO DOS DADOS###########################################
#==============================================================================================================

# Expressões regulares para limpeza dos dados
def normalize_text(df, text_field):
    """Aplica a função Regular Expression para normalizar o texto do DataFrame

    Args:
        df ([Pandas DataFrame]): Dataframe que possue o campo com os comentários
        text_field (String): Coluna do DataFrame com os comentários a serem normalizados

    Returns:
        String: Comentários normalizados
    """    
    # remove tags de quebra de linha e carriage return
    df[text_field] = df[text_field].str.replace(r"[\n\r]", " ")
    # remove links
    df[text_field] = df[text_field].str.replace((r"http\S+"), "")
    # remove a palavra http
    df[text_field] = df[text_field].str.replace(r"http", "")
    # remove qualquer texto que comece com arroba
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    # Substitui 3 ou mais letras consecutivas por 1. maravilhoooooso -> maravilhoso
    df[text_field] = df[text_field].str.replace(r"(.)\1\1+", r"\1")
    # Remove datas
    df[text_field] = df[text_field].str.replace("(\d+(/|-){1}\d+(/|-){1}\d{2,4})", "")
    # Remove números
    df[text_field] = df[text_field].str.replace('\d+', '')
    # Normaliza o texto deixando todas as palavras com caracteres minúsculos
    df[text_field] = df[text_field].str.lower()
    # Remove acentos
    df[text_field] = df[text_field].map(lambda c: normalize('NFKD', c)).str.encode('ASCII', 'ignore').str.decode('ASCII')
    return df

# Remove pontuação
def remove_punctuations(text):
    # Lista os carcters de pontuação
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text

def tokenizer(row):
    """ Este tokenizer irá quebrar o texto e criar uma lista de palavras (bag of words). 
        Será utilizada a o word_tokenizer da biblioteca NLTK
    """
    review = row['review']
    tokens = word_tokenize(review)
    return tokens

def remove_stops(row):
    """Remove as palvras da coluna tokens que estão presente na lista stopwords
    """
    # Lista customizada de stopwords.
    custom_stops = [
        'lannister', 'tyrell', 'arryn', 'targaryen', 'baratheon', 'tully',
        'greyjoy', 'martell', 'stark', 'americanas', 'loja', 'lojas', 'produto',
        
    ]
    # lista padrão de stopwords em portugues da biblioteca NLTK
    stops = list(stopwords.words('portuguese'))
    # acrescenta as casas de Game of Thrones na lista padrão
    stops.extend(custom_stops)
    # set -> valores únicos e  mais rápido do que a lista quando se trata de determinar se um objeto está presente
    stop_words = set(stops)
    # retira a palavra "não" da lista de stopwords para manter o sentido de negação em bigrams
    stop_words.remove('não')
    # retira a palavra "muito" da lista de stopwords para manter o sentido de intensidade
    # me bigrams
    stop_words.remove('muito')

    # script para aplicar a remoção a cada linha do DataFrame
    my_list = row['tokens']
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)

def rejoin_words(df, col_name, text_field):
    """Reúne os tokens em uma string única

    Args:
        df (Pandas DatFrame): dataframe com o dataset
        col_name (string): nome da coluna que receberá o output da função
        text_field (string): nome da coluna onde será aplicada a função

    Returns:
        Pandas DataFrame: retorna o dataframe com a nova coluna gerada pela função
    """        
    df[col_name] = df[text_field].str.join(sep=' ')
    return df

#latex label: cod:FuncExtractFeaturesFromCorpus
def extract_features_from_corpus(corpus, vectorizer, n_rows=5, n_cols=10, df=False):
    """Gera um DataFrame de exemplo da Matrix de Contagem de Tokens
        e retorna o resultado do método fit_tranform do vecotrizer para uma variável

    Args:
        corpus (str): texto em lista com corpus
        vectorizer (metodo python): metodo que ira tranformar o corpus em uma matriz
        n_rows (int, optional): número de linha do DataFrame de exemplo. Defaults to 5.
        n_cols (int, optional): número de colunas do DataFrame de exemplo. Defaults to 10.
        df (bool, optional): opção de gerar um DataFrame de exemplo. Defaults to False.

    Returns:
        sparse matrix: Aprenda o dicionário de vocabulário e retorne a matriz de termos de
        documentos.
        pandas dataframe(opcional): Retorna uma DataFrame com um exemplo da matriz
    """
    
    # Features Extraction
    corpus_features = vectorizer.fit_transform(corpus)
    features_names = vectorizer.get_feature_names()
    
    # Cria um dataframe para demonstrar o vetor criado no processo
    df_corpus_features = None
    if df:
        # Exemplo do vetor de palavras
        df_corpus_features = pd.DataFrame(corpus_features.toarray()[:n_rows,:n_cols], columns=features_names[:n_cols])
        # Inicia o index com 1
        df_corpus_features.index +=1
        # Acrescenta o prefixo "doc_" no index
        df_corpus_features.set_index('doc_' + df_corpus_features.index.astype(str), inplace=True)
    
    return corpus_features, df_corpus_features

#latex label: cod:FuncPlotGaussian
def plot_gaussian(df, min_lim=-3, max_lim=3, savefig=False):
    """Plota a curva da função de probabilidade de densisdade
    para cada classe de forma independente. 

    Args:
        df (Pandas DataFrame): Dataframe com uma coluna de features e uma de target
        min_lim (int, optional): limite inferior do eixo x (nº de desvios-padrão). Defaults to -3.
        max_lim (int, optional): limite superior do eixo x (nº de desvios-padrão). Defaults to 3.
    """
    from scipy import stats

    # métricas da classificação positiva
    df_pos = df[df.iloc[:,1] == 1]
    mean_pos = df_pos.iloc[:,0].mean()
    std_pos = df_pos.iloc[:,0].std()
    y_values_pos = stats.norm(mean_pos, std_pos)
    
    # métricas da classificação negativa
    df_neg = df[df.iloc[:,1] == 0]
    mean_neg = df_neg.iloc[:,0].mean()
    std_neg = df_neg.iloc[:,0].std()
    y_values_neg = stats.norm(mean_neg, std_neg)
    
    # Cria um objeto matplolib
    fig, ax = plt.subplots(figsize=(9,6))
    
    # Valores do eixo x
    x_values = np.linspace(min_lim, max_lim, 120)
    
    # Especifica a região a preencher da cruva gaussiana
    x_fill = np.arange(min_lim, max_lim, 0.001)
    # Curva positiva
    y_fill_pos = stats.norm.pdf(x_fill, mean_pos, std_pos)
    ax.fill_between(x_fill,y_fill_pos,0, alpha=0.2, color='blue')
    ax.plot(x_values, y_values_pos.pdf(x_values), linewidth=1)

    # Curva negativa
    # pdf = Função de Densidade de Probabilidade
    y_fill_neg = stats.norm.pdf(x_fill, mean_neg, std_neg)
    ax.fill_between(x_fill,y_fill_neg,0, alpha=0.2, color='red')
    ax.plot(x_values, y_values_neg.pdf(x_values), linewidth=1)

    plt.legend(['Positivo (1)', 'Negativo (0)'])

    if savefig == True:
        plt.savefig(f'out/gaussian_{dt.date.today()}.png', dpi=300, pad_inches=0, bbox_inches = 'tight')
    plt.tight_layout()
    plt.show()

#==============================================================================================================
######################################## CRIAÇÃO E AVALIAÇÃO DO MODELO#########################################
#==============================================================================================================

def make_custom_sample(df_corpus, size, p=0.304):
    """Cria uma amostra do corpus matenndo a mesma 
    proporção de distribuição entre as classes.

    Args:
        df_corpus (Pandas DataFramae): dataframe com o corpus
        size (int): quantidade de registros da amostra
        p (float, optional): porcetagem de de registros da classe minoritária. Defaults to 0.304.

    Returns:
        Pandas DataFrame: amostra do corpus com o número de registros definidos 
    """     

    neg = df_corpus[df_corpus['score'] == 0].sample(int(size * p), random_state=42)
    pos = df_corpus[df_corpus['score'] == 1].sample(int(size * (1-p)), random_state=42)

    df = pd.concat([neg, pos]).reset_index(drop=True)
    return df

def custom_cf_matrix(true_labels, predicted_labels, normalize='true', data_name='', title_size=14, cmap='Blues', ax=None, savefig=False):
    """Esta função é usada para traçar e personalizar uma matriz de confusão para um modelo específico.

    Args:
        y_test (np.array): variável de destino de teste a ser usada na avaliação
        y_pred (np.array): matriz de previsões fornecidas pelo respectivo modelo
        data_name (str, optional): nome do dataset . Defaults to ''.
        title_size (int, optional): Tamanho do título. Defaults to 14.
        cmap (str, optional): [nome da cor]. Defaults to 'Blues'.
        normalize (‘true’, ‘pred’, ‘all’). Default=None. Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. If None, confusion matrix will not be normalized.)
    """    

    # Calcula plot a Confusion matrix com os dados normalizados sobre as linhas verdadeiras
    cf_matrix = metrics.confusion_matrix(true_labels, predicted_labels, normalize=normalize)
    plt.imshow(cf_matrix, aspect=0.75)

    categories  = ['Negativo','Positivo']
    group_names = ['V.N.','F.P.', 'F.N.','V.P.']
    
    if normalize == 'true':
        group_values = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()]
    else:
        group_values = ['{0:}'.format(value) for value in cf_matrix.flatten()]
    
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_values)]
    labels = np.asarray(labels).reshape(2,2)

    
    _ = sns.heatmap(cf_matrix, annot = labels, annot_kws={"size": 18},
                cmap = cmap, fmt = '', square=None, cbar=False,
                xticklabels = categories, yticklabels = categories)
    _.set_xticklabels(_.get_xmajorticklabels(), fontsize = 12)
    _.set_yticklabels(_.get_ymajorticklabels(), fontsize = 12)

    plt.xlabel("Valores Preditos", fontdict = {'size':16}, labelpad=10)
    plt.ylabel("Valores Reais"   , fontdict = {'size':16}, labelpad=10)
    plt.title (f"{data_name} ", fontdict = {'size':title_size}, pad=10)

    if savefig == True:
        plt.savefig(f'out/custom_cf_maxtrix_{dt.date.today()}.png', dpi=300, pad_inches=0, bbox_inches = 'tight')

def custom_cf_matrix_from_dict(predicted_labels, true_labels,  normalize='true', title_size=14, cmap='Blues', savefig=False):
    """Esta função é usada para traçar e personalizar uma matriz de confusão para um modelo específico.

    Args:
        y_test (np.array): variável de destino de teste a ser usada na avaliação
        y_pred (np.array): matriz de previsões fornecidas pelo respectivo modelo
        data_name (str, optional): nome do dataset . Defaults to ''.
        title_size (int, optional): Tamanho do título. Defaults to 14.
        cmap (str, optional): [nome da cor]. Defaults to 'Blues'.
        normalize (‘true’, ‘pred’, ‘all’). Default=None. Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. If None, confusion matrix will not be normalized.)
    """    
    plt.figure(figsize=(10, 4))
    k = 1

    for model_name, label_pred in predicted_labels.items():
        # Constrói a Matriz de Confusão
        plt.subplot(1, 3, k)
        plt.tight_layout()
        custom_cf_matrix(true_labels=true_labels, predicted_labels=label_pred, normalize=normalize, data_name=model_name, title_size=title_size, cmap=cmap)
        k += 1

    if savefig == True:
        plt.savefig(f'out/custom_cf_maxtrix_{dt.date.today()}.png', dpi=300, pad_inches=0, bbox_inches = 'tight')
    
    plt.tight_layout()
    plt.show()

def plot_cross_validation(k=5, savefig=False):
    """Função adaptada do e-book Andreas C. Müller and Sarah Guido. “Introduction to
    Machine Learning with Python”. Gera uma figura de exemplo de uma k-fold cross-validation

    Args:
        k (int, optional): [description]. Defaults to 5. Número de folds do exemplo.
        savefig (bool, optional): [description]. Defaults to False. Define se salva a figura no disco.
    """
    plt.figure(figsize=(12, int(k/2)))
    plt.title(f"{k}-fold Cross Validation")
    axes = plt.gca()
    axes.set_frame_on(False)

    n_folds = k
    n_samples = 25

    n_samples_per_fold = n_samples / float(n_folds)

    for i in range(n_folds):
        colors = ["w"] * n_folds
        colors[i] = "blue"
        bars = plt.barh(
            y=range(n_folds), width=[n_samples_per_fold - 0.1] * n_folds,
            left=i * n_samples_per_fold, height=.6, color=colors, hatch="//",
            edgecolor='k', align='edge')
    axes.invert_yaxis()
    axes.set_xlim(0, n_samples + 1)
    plt.ylabel("Interações com CV")
    plt.xlabel("Dados")
    plt.xticks(np.arange(n_samples_per_fold / 2., n_samples,
                         n_samples_per_fold),
               ["Fold %d" % x for x in range(1, n_folds + 1)])
    plt.yticks(np.arange(n_folds) + .3,
               ["Split %d" % x for x in range(1, n_folds + 1)])
    plt.legend([bars[0], bars[k-1]], ['Dados de Treino', 'Dados de Teste'],
               loc=(0.97, 0.4), frameon=False)

    if savefig == True:
        plt.savefig(f'out/cross_val_example_{dt.date.today()}.png', dpi=300, pad_inches=0, bbox_inches = 'tight')

def plot_binary_confusion_matrix(savefig=False):
    #Função adaptada do e-book Andreas C. Müller and Sarah Guido. “Introduction to Machine Learning with Python”.
    sns.set_style('white')
    plt.text(0.45, .6, "VN", size=100, horizontalalignment='right')
    plt.text(0.45, .1, "FN", size=100, horizontalalignment='right')
    plt.text(.95, .6, "FP", size=100, horizontalalignment='right')
    plt.text(.95, 0.1, "VP", size=100, horizontalalignment='right')
    plt.xticks([.23, .73], ["negativo predito", "positivo predito"], size=14)
    plt.yticks([.40, .95], ["classe positiva", "classe negativa"], size=14, rotation=90)
    plt.plot([.5, .5], [0, 1], '--', c='k')
    plt.plot([0, 1], [.5, .5], '--', c='k')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if savefig == True:
        plt.savefig(f'out/cf_matrix_example_{dt.date.today()}.png', dpi=300, pad_inches=0)

def plot_auc_from_dict(pred_proba_labels, true_labels, figsize=(7,5), savefig=False):

    plt.subplots(figsize=figsize)
    plt_linestyles = ['dotted', 'dashed', 'dashdot']
    plt_linecolor = ['royalblue', 'firebrick', 'darkcyan']
    i = 0
    
    #AUC Curve
    for clf_name, y_pred_proba in pred_proba_labels.items():
        
        # Calcula a TFP, TVP e o limite de corte (thr)
        fpr, tpr, thr = metrics.roc_curve(true_labels,  y_pred_proba)
        # Cacula a área sob a curva ROC
        auc = metrics.roc_auc_score(true_labels, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{clf_name}: {auc:.2%}', linewidth=3., linestyle=plt_linestyles[i], color=plt_linecolor[i])
        i+=1
        #ax.fill_between(fpr, tpr, 0, alpha=i, color='purple')


    plt.plot([0, 1], [0, 1], 'k--', alpha=0.8)
    plt.legend(loc=4,  prop={'size': 12})
    plt.xlim([-0.02, 1.])
    plt.ylim([0.0, 1.02])
    plt.xlabel('Taxa de Falso Positvos (TFP)', size=12, labelpad=10)
    plt.ylabel('Taxa de Verdadeiro Posivitos (TVP)', size=12, labelpad=10)
    if savefig == False:
        plt.title('Curva AUC', size=15, pad=10)
    if savefig == True:
        plt.savefig(f'out/roc_curve_from_dict{dt.date.today()}.png', dpi=300, pad_inches=0,bbox_inches = 'tight')
    plt.tight_layout()
    plt.show()

def plot_auc(clf, features, true_labels,  predicted_labes, savefig=False):
    # Accuracy
    print(f'Acurácia = {metrics.accuracy_score(true_labels, predicted_labes):.2%}\n')
    
    fig, ax = plt.subplots(figsize=(7,5))
    #AUC Curve
    y_pred_proba = clf.predict_proba(features)[::,1]
    fpr, tpr, thr = metrics.roc_curve(true_labels,  y_pred_proba)
    auc = metrics.roc_auc_score(true_labels, y_pred_proba)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.8)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2%}', color='grey')
    ax.fill_between(fpr, tpr, 0, alpha=0.2, color='purple')
    plt.legend(loc=4,  prop={'size': 12})
    plt.xlim([-0.01, 1.])
    plt.ylim([0.0, 1.])
    plt.xlabel('Taxa de Falso Positvos (TFP)', size=12, labelpad=10)
    plt.ylabel('Taxa de Verdadeiro Posivitos (TVP)', size=12, labelpad=10)
    if savefig == False:
        plt.title('Curva AUC', size=15, pad=10)
    if savefig == True:
        plt.savefig(f'out/roc_curve_{dt.date.today()}.png', dpi=300, pad_inches=0,bbox_inches = 'tight')
    plt.tight_layout()
    plt.show()

def train_get_metrics(approach, classifier, train_features, test_features, train_labels, test_labels, cv=5, predictions=False):
    
    t0 = time.time()
    clf = classifier
    clf.fit(train_features, train_labels)

    cv_train_scores = cross_val_score(clf, train_features, train_labels, scoring='f1_weighted', cv=cv, n_jobs=-1)
    cv_train_mean_score = np.mean(cv_train_scores)
    labels_pred = clf.predict(test_features)
    # Testa se o algoritmo calcula probabilidade com predict_proba ou decision_function
    try:
        pred_proba_labels = clf.predict_proba(test_features)[::,1]
    except:
        pred_proba_labels = clf.decision_function(test_features)

    cv_test_score = metrics.f1_score(test_labels, labels_pred, average='weighted')
    t1 = time.time()
    delta_time = t1 - t0

    model_list = []
    model_performance = {}
    model_performance['modelo'] = clf.__class__.__name__
    model_performance[f'Medida-F de Treino ({approach})'] = '{:.4f}'.format(cv_train_mean_score)
    model_performance[f'Medida-F de Teste ({approach})'] = '{:.4f}'.format(cv_test_score)
    model_performance[f'Tempo Total({approach})'] = round(delta_time, 3)
    model_list.append(model_performance)
    df = pd.DataFrame(model_list)

    if predictions == True:
        return labels_pred, pred_proba_labels, df
    else:
        return df

def tf_tfidf_dataframe_metrics(tf_dataframe, tfidf_dataframe):
    clf_performance = tf_dataframe.merge(tfidf_dataframe, on='modelo')
    print('Desempenho do Modelo:')
    display(clf_performance.T)

    return clf_performance

def grid_search_cv(pipe, param_grid, train_features, train_labels, cv=5):
    t0 = time.time()
    grid = GridSearchCV(pipe, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1)
    grid.fit(train_features, train_labels)
    t1 = time.time()
    delta_time = round((t1 - t0) / 60)

    print(f'Modelo: {pipe[1].__class__.__name__}')
    print(f'Tempo gasto: {delta_time} minutos.')
    print(f'Melhor nota do cross-validation: {grid.best_score_:.2%}')
    print(f'Melhor parâmetro: {grid.best_params_}')

    return grid

def evaluate_model(true_labels, predicted_labels, predicted_proba_labels):
    test_list = [] # Dataframe list

    # Performance do Modelo sobre os dados de teste
    tn, fp, fn, tp = metrics.confusion_matrix(true_labels, predicted_labels).ravel()
    specificity = tn / (tn+fp)

    accuracy_test = metrics.accuracy_score(true_labels, predicted_labels)
    precision_test = metrics.precision_score(true_labels, predicted_labels)
    recall_test = metrics.recall_score(true_labels, predicted_labels)
    f1_test = metrics.f1_score(true_labels, predicted_labels)
    auc_test = metrics.roc_auc_score(true_labels, predicted_proba_labels[:,1])

    # Criando o DataFrame com as métricas de teste
    test_performance = {}
    test_performance['acurácia'] = accuracy_test
    test_performance['precisão'] = precision_test
    test_performance['revocação'] = recall_test
    test_performance['f1_score'] = f1_test
    test_performance['especificidade'] = specificity
    test_performance['auc'] = auc_test

    # Salva em uma lista o resultado de avaliação do modelo
    test_list.append(test_performance)

    # DataFrame com as métricas de teste
    df = pd.DataFrame(test_list)

    # for column in ['acurácia', 'precisão', 'revocação', 'f1_score', 'especificidade', 'auc']:
    #     df[column] = df[column].map('{:.2%}'.format)
    for column in df.columns:
        df[column] = df[column].map('{:.2%}'.format)

    return df

def interpret_classification_model_prediction(clf, doc_index, test_corpus, true_labels, explainer_obj, text=False, save_to_file=False):
    # mostra as predições e o sentimento real do modelo
    classes = list(clf.classes_)
    print(
        f'''
        Index do documento de teste: {doc_index}
        Sentimento real: {true_labels[doc_index]}
        Sentimento predito: {clf.predict([test_corpus[doc_index]])}
'''
    )
    
    # mostra as probabilidades das predições
    print(f'\nProbabilidade de predição do Modelo:')
    for probabilities in zip(classes, clf.predict_proba([test_corpus[doc_index]])[0]):
        print(f'Classe {probabilities[0]}: {round(probabilities[1], 6)}')
    # display model prediction interpretation
    exp = explainer_obj.explain_instance(test_corpus[doc_index], 
    clf.predict_proba, num_features=10, 
    labels=[1])
    exp.show_in_notebook(text=text)
    if save_to_file == True:
        exp.save_to_file(f'out/doc_{doc_index}_explanation.html')

#==============================================================================================================
################################################# PIPELINE ###################################################
#==============================================================================================================
def make_corpus_dataframe(text):
    """Cria uma dataframe a partir da tupla (texto de avaliação, classificação da
avaliação, zerou ou um)

    Args:
        text (tupla): tupla composta pela avaliação de texto e classificação. A ordem da
        tupla deve ser respeitada (texte, classificação)

    Returns:
        Pandas Dataframe: datafram composto pela avaliação e classificação 
        com o layout necessário para se aplicar o pipeline de processamento do texto.
    """
    if type(text) is not list:
        text = [text]
    df = pd.DataFrame(data=text, columns=['review', 'score'])
    return df

def normalize_corpus(text):
    "Pipeline com todas as funções necessárias para processar o texto"

    data = text
    data = normalize_text(data, 'review')
    data['review'] = data['review'].apply(remove_punctuations)
    data.drop(data[data['review'].map(len) <= 2].index, inplace=True)
    data['tokens'] = data.apply(tokenizer, axis=1)
    # Cria uma nova coluna no datataset (imporant_tokens) com os termos que não esta na lista de stopwords
    data['important_tokens'] = data.apply(remove_stops, axis=1)
    # Cria uma nova coluna com os tokens reunidos em uma string única
    data = rejoin_words(data, 'processed_review', 'important_tokens')
    # elimina colunas que não são mais necessárias
    data.drop(['review', 'tokens', 'important_tokens'], axis=1, inplace=True)
    # Remove as linhas com 2 ou menos caractere
    data.drop(data[data['processed_review'].map(len) <= 2].index, inplace=True)
    # Coloca as colunas em uma ordem mais padronizada (label por último)
    data = data[['processed_review', 'score']]

    return data



# PRETTIFY

def module_sucsess():
    print(f'Módulo importado com sucesso em {dt.datetime.now().strftime("%d-%m-%Y às %H:%M:%S")}')
module_sucsess()

def success_message(notebook):

    notebook = notebook

    if notebook == 1:
        print(
            f''' 
    ========================================================================================================

    Todos os trechos de código deste Jupyter Notebook foram executados com sucesso.
    O corpus limpo e processado foi salvo na pasta do projeto com o nome corpus_{dt.date.today()}.csv.
    Agora prossiga no Jupyter Notebook 02_Eploratory_Analysis.ipynb para a Análise Exploratória dos Dados.
    
    ========================================================================================================
    '''
        )
    
    if notebook == 2:
        print(
            f''' 
    ========================================================================================================

    Todos os trechos de código deste Jupyter Notebook foram executados com sucesso.
    A análise exploratória dos dados foi encerrada.
    Agora prossiga no Jupyter Notebook 03_Modelos_e_Metricas.ipynb para Modelagem e
    Avaliadção Modelo .
    
    ========================================================================================================
    '''
        )
    if notebook == 3:
        print(
            f''' 
    ========================================================================================================

    Todos os trechos de código deste Jupyter Notebook foram executados com sucesso.
    Foram realizados o treinamento, ajustes e escolha do modelo.
    Agora prossiga no Jupyter Notebook 04_ModeloFinal.ipynb para
    Avaliadção Modelo Escolhido.
        
    ========================================================================================================
    '''
        )
    if notebook == 4:
        print(
            f''' 
    ========================================================================================================

    Todos os trechos de código deste Jupyter Notebook foram executados com sucesso.
    Chegamos ao fim. Caso queira executar o modelo em novos dados, prossiga para o Jupyter
    Notebook 05_Novos_Dados.ipynb
        
    ========================================================================================================
    '''
        )
