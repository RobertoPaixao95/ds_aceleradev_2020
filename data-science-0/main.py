#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday


# In[4]:


# Questão 1
# Shape do Dataset

obs_col = black_friday.shape
obs_col


# In[35]:


# Questão 2
# Query procurando por gênero Feminino e idades entre 26-35 anos

mulheres_26_35 = len(black_friday.query('Gender == "F" & Age == "26-35"'))
mulheres_26_35


# In[6]:


# Questão 3
# Quantos usuários únicos há no dataset?

usuarios_unicos = black_friday['User_ID'].nunique()
usuarios_unicos


# In[7]:


# Questão 4
# Quantos tipos de dados diferentes há no dataset?

dif_dados = black_friday.dtypes.nunique()
dif_dados


# In[8]:


# Questão 5
# Qual porcentagem dos registros possui ao menos um valor null (None, ǸaN etc)?

total_null = max(black_friday.isnull().sum())
total_linhas = black_friday.shape[0]

porcentagem_faltante = total_null / total_linhas

porcentagem_faltante


# In[9]:


# Questão 6
# Quantos valores null existem na variável (coluna) com o maior número de null?

valores_null = max(black_friday.isnull().sum())
valores_null


# In[24]:


# Questão 7
# Qual o valor mais frequente (sem contar nulls) em Product_Category_3?

val_freq = int(black_friday['Product_Category_3'].mode())
val_freq


# In[ ]:


# Questão 8
# Qual a nova média da variável (coluna) Purchase após sua normalização?

x = black_friday.Purchase.values
x_min = black_friday.Purchase.min()
x_max = black_friday.Purchase.max()

x_norm = (x - x_min)/(x_max - x_min)
x_media = float(x_norm.mean())
x_media


# In[ ]:


# Questão 9
# Quantas ocorrências entre -1 e 1 inclusive existem da variável Purchase após sua padronização?

x = black_friday.Purchase.values
media = black_friday.Purchase.mean()
desvio_padrao = black_friday.Purchase.std()

padronizado = (x - media) / desvio_padrao
ocorrencia = len([x for x in padronizado if x > -1 and x < 1])
ocorrencia


# In[ ]:


# Questão 10
# Podemos afirmar que se uma observação é null em Product_Category_2 ela também o é em Product_Category_3?

# Checando a quantidade de valores nulos de Product_Category_2
p2_null = len(black_friday.query('Product_Category_2 == "NaN"'))

# Checando a quantidade de valores nulos de Product_Category_2 e Product_Category_3
p2_p3_null = len(black_friday.query('Product_Category_2 == "NaN" and Product_Category_3 == "NaN"'))
resultado = bool(p2_p3_null)
resultado


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return obs_col
    


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return mulheres_26_35
    


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return usuarios_unicos


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return dif_dados


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[ ]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return porcentagem_faltante


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return valores_null


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return val_freq


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return x_media


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    # Retorne aqui o resultado da questão 9.
    return ocorrencia


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[ ]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return resultado

