#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[ ]:


countries = pd.read_csv("countries.csv")


# In[ ]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[ ]:


countries.dtypes


# In[ ]:


col_filter_objects = countries.drop(columns=['Country', 'Population', 'Area', 'Region', 'GDP']).columns.tolist()


# In[ ]:


col_filter_objects


# In[ ]:


countries[col_filter_objects] = countries[col_filter_objects].apply(lambda feature: feature.str.replace(',','.').astype('float'))


# In[ ]:


countries.head()


# In[ ]:


countries.dtypes


# In[ ]:


countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[ ]:


def q1():
    unique_regions = countries['Region'].sort_values().unique().tolist()
    return unique_regions


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[ ]:


def q2():
    kbins_discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy='quantile')
    kbins_discretizer.fit(countries[['Pop_density']])
    pop_density_discretized = kbins_discretizer.transform(countries[['Pop_density']])
    
    return int(sum(pop_density_discretized >= 9))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[ ]:


def q3():
    ohe = OneHotEncoder()
    countries['Climate'] = countries['Climate'].fillna(countries['Climate'].mean())

    region_ohe_sum = ohe.fit_transform(countries[['Region']])
    climate_ohe_sum = ohe.fit_transform(countries[['Climate']])
    return int(climate_ohe_sum.shape[1] + region_ohe_sum.shape[1])


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[ ]:


features_int_float = countries.select_dtypes(['int64', 'float64']).columns


# In[ ]:


# 1.
# countries[features_int_float] = countries[features_int_float].fillna(countries[features_int_float].median())

# 2.
# std_scaler = StandardScaler()
# countries[features_int_float] = std_scaler.fit_transform(countries[features_int_float])


# In[ ]:


pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])


# In[ ]:


pipe.fit_transform(countries[features_int_float])


# In[ ]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[ ]:


def q4():
    test_country_transform = pipe.transform([test_country[2:]])
    df_test_country = pd.DataFrame(test_country_transform, columns=features_int_float)
    
    return float(round(df_test_country.Arable,3))    


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[ ]:


# figsize(15,5)
# sns.boxplot(countries.Net_migration)


# In[ ]:


def q5():
    net_migration = countries.Net_migration.copy()
    Q1 = net_migration.quantile(0.25)
    Q3 = net_migration.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit, upper_limit = Q1 - (1.5 * IQR), Q3 + (1.5 * IQR)
    outlier_abaixo = len(net_migration[net_migration < lower_limit])
    outliers_acima = len(net_migration[net_migration > upper_limit])
    total_outliers = outlier_abaixo + outliers_acima
    missed_data = total_outliers / net_migration.shape[0]
    
    return outlier_abaixo, outliers_acima, missed_data <= 0.15


# In[ ]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[ ]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[ ]:


def q6():
    
    count_vectorizer = CountVectorizer()
    newsgroup_count = count_vectorizer.fit_transform(newsgroup.data)
    phone = count_vectorizer.vocabulary_.get('phone'.lower())
    phone_count = newsgroup_count[:,phone].sum()

    return int(phone_count)


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[ ]:


def q7():
    tf_vectorizer = TfidfVectorizer()
    newsgroup_vectorized = tf_vectorizer.fit_transform(newsgroup.data)
    phone = tf_vectorizer.vocabulary_.get('phone'.lower())
    phone_tf_idf = float(round(newsgroup_vectorized[:,phone].sum(),3))
    return phone_tf_idf

