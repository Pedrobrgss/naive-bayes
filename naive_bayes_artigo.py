from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

# Configurar opção do Pandas para evitar problemas futuros
pd.set_option('future.no_silent_downcasting', True)

# Carregar a base de dados
df = pd.read_excel("C:/Users/Pedro/Desktop/IA/Dados do Teste .xlsx")

# Substituir '?' por NaN e garantir que os tipos sejam inferidos corretamente
df.replace(['?', pd.NA], np.nan, inplace=True)
df = df.infer_objects(copy=False)

# Definir a variável alvo e as variáveis preditoras
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]

# Identificar colunas numéricas e categóricas
numeric_features = ["A1 - Score", "A2 - Score", "A3 - Score", "A4 - Score", "A5 - Score", 
                    "A6 = Score", "A7 - Score", "A8 - Score", "A9 - Score", "A10 - Score",
                    "Age Numeric", "Result Numeric"]

categorical_features = ["Gender", "Ethnicity", "Jundice", "Autism", "Country of Residence", 
                        "Used App Before", "Age Description", "Relation"]

# Criar transformadores para pré-processamento
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Criar transformador de colunas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# Ajustar o pré-processador antes do train_test_split para evitar problemas de categorias desconhecidas
X_processed = preprocessor.fit_transform(X)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Criar pipeline do modelo
model_pipeline = Pipeline(steps=[
    ('classifier', GaussianNB())
])

# Treinar o modelo
model_pipeline.fit(X_train, y_train)

# Avaliar o modelo
accuracy = model_pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")