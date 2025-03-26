# 🧠 Classificação de Autismo com Naive Bayes

Este projeto utiliza **Naive Bayes** para prever a presença de **Transtorno do Espectro Autista (TEA)** com base em um conjunto de dados clínicos.  
O modelo é treinado e avaliado usando **Python, Scikit-Learn e Pandas**.  

---

## 📌 **Como funciona?**

### 1️⃣ **Pré-processamento de dados**
O conjunto de dados contém informações sobre pacientes, incluindo:
- **Respostas a perguntas de diagnóstico** (valores numéricos de 1 a 10)
- **Informações demográficas** (gênero, país de residência, histórico familiar, etc.)
- **Idade e resultado do teste de autismo**

### 2️⃣ **Modelo de Machine Learning**
Utilizamos o **Naive Bayes Gaussiano**, que é eficiente para **dados contínuos e categóricos**.  
- **Princípio**: Baseia-se na probabilidade condicional para prever se um indivíduo tem ou não **TEA**.  
- **Por que Naive Bayes?**  
  - Rápido e eficiente  
  - Funciona bem com **dados ruidosos**  
  - Boa generalização para **pequenos conjuntos de dados**  

---

## 📊 **Resultados**
Após o treinamento, o modelo alcançou uma precisão aproximada de **31.21%**.

---

## 🔬 **O que é Naive Bayes?**
O **Naive Bayes** é um algoritmo de classificação baseado no **Teorema de Bayes**:

$$
P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}
$$

Ele assume que **todas as variáveis são independentes entre si**, o que raramente é verdade na prática, mas ainda assim funciona muito bem para muitas aplicações.
---

## 🛠 **Tecnologias utilizadas**
- **Python 3.12**
- **Scikit-Learn** (para Machine Learning)
- **Pandas & NumPy** (para manipulação de dados)
- **Jupyter Notebook** (para experimentação)

---

## 📌 **Possíveis melhorias**
🔹 Testar outros modelos como **Árvores de Decisão ou SVM**  
🔹 Melhorar o tratamento de **valores ausentes**  
🔹 Ampliar o conjunto de dados para maior generalização  

---
🚀 **Desenvolvido por [Pedro Borges Alves](https://github.com/Pedrobrgss)**  
