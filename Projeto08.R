# Projeto com Feedback 8 - Modelagem Preditiva em IoT
# Previsão de Uso de Energia

# Rosane Moreira Barbosa

# Objetivo: 
# Criação de modelos preditivos para a previsão de consumo de energia de
# eletrodomésticos. Os dados utilizados incluem medições de sensores de 
# temperatura e umidade de uma rede sem fio, previsão do tempo de uma estação
# de um aeroporto e uso de energia utilizada por luminárias.
# 
# O conjunto de dados foi coletado por períodos de 10 minutos por cerca de
# 5 meses.
#
# *** Descrição das Variáveis ***
# Appliances -> Uso de energia dos eletrodomésticos (suposição: variável target)
# lights -> Potência da luminária
# T -> Temperatura
# T_out -> Temperatura fora da casa
# RH -> Umidade Relativa (Relative Humidity)
# RH_out -> Umidade fora da casa
# Press_mm_hg -> Unidade de pressão
# Tdewpoint -> medida do teor de vapor de água em um gás
# NSM -> North Star Metric(pesquisa Google: métrica que melhor captura o principal valor do seu produto ou serviço)
# rv1 e rv2 -> RV battery / variáveis aleatórias



# Definindo o diretório de Trabalho
setwd("C:/rosane/FCD-DSA/MachineLearning/Cap20/Projeto08")
getwd()

# Carregando os Pacotes
library(plyr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(randomForest)
library(stats)
library(caret)  # pacote de ML
library(e1071)  # pacote que contém o algoritmo SVM
library(ModelMetrics)
library(xgboost)


#***********************************************
#** Carregando os Dados de Treino e de Teste ***
#***********************************************

# Carregando os dados de treino
df_iot_treino <- read.csv("projeto8-training.csv", header = TRUE, sep = ',')

# Dimensão do arquivo de dados (14803 observações e 32 variáveis)
dim(df_iot_treino)

# Visualizando os dados
View(df_iot_treino)


# Carregando os dados de teste
df_iot_teste <- read.csv("projeto8-testing.csv", header = TRUE, sep = ',')
dim(df_iot_teste)
View(df_iot_teste)


# Concatenando os dados de treino e teste em um único dataframe a fim de analisar
# os dados em conjunto e fazer um pré-processamento único de todos os dados.

df_iot <- rbind(df_iot_treino, df_iot_teste)

dim(df_iot)


# Removendo os datasets de treino e teste para liberar a memória
rm(df_iot_treino)
rm(df_iot_teste)


#***********************************************
#******* Análise Exploratória dos Dados ********
#***********************************************

# Verificando os tipos dos dados
str(df_iot)

# Verificando se existem valores missing
sapply(df_iot, function (x) sum(is.na(x)))

# Resumo das medidas de tendência central das variáveis 
summary(df_iot)


#***** Boxplots *****

# Vetor das variáveis de Temperatura
Tx <- c("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T_out")

# Loop para os boxplots da temperatura
# Comentários: 
# Verifica-se que algumas medidas de temperatura apresentam outliers
# T6 e T_out apresentam temperaturas negativas
for (x in Tx) {boxplot(df_iot[[x]], main = "Boxplot da Temperatura", ylab = x)}

# Vetor das variáveis de umidade
RH_x <- c("RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9", "RH_out")

# Loop para os boxplots da umidade
# Comentários:
# Verifica-se que algumas medidas de umidade relativa apresentam outliers
# A umidade de RH_6 e RH_out apresentam uma variação maior.
for (x in RH_x) {boxplot(df_iot[[x]], main = "Boxplot da Umidade", ylab = x)}


# Comentário sobre o boxplot da variável target:
# No boxplot da variável target (Appliances), abaixo, Verifica-se que 
# existem vários outliers e há um valor muito discrepante (1080)
boxplot(df_iot$Appliances, main = "Boxplot dos Eletrodomésticos - IoT",
        col.main = "red",
        col = "red",
        ylab = "Appliances", 
        horizontal = TRUE)

boxplot(df_iot$Press_mm_hg, main = "Boxplot da Unidade de Pressão",
        col.main = "red",
        col = "red",
        ylab = "Press_mm_hg", 
        horizontal = TRUE)

boxplot(df_iot$Windspeed, main = "Boxplot da Velocidade do Vento",
        col.main = "red",
        col = "red",
        ylab = "Windspeed", 
        horizontal = TRUE)

boxplot(df_iot$Visibility, main = "Boxplot da Visibilidade",
        col.main = "red",
        col = "red",
        ylab = "Visibility", 
        horizontal = TRUE)

boxplot(df_iot$Tdewpoint, main = "Boxplot do Ponto de Condensação da Água",
        col.main = "red",
        col = "red",
        ylab = "Tdewpoint", 
        horizontal = TRUE)

boxplot(df_iot$NSM, main = "Boxplot de North Star Metric",
        col.main = "red",
        col = "red",
        ylab = "NSM", 
        horizontal = TRUE)

boxplot(df_iot$rv1, main = "Boxplot de rv1",
        col.main = "red",
        col = "red",
        ylab = "rv1", 
        horizontal = TRUE)


 
#***** Histogramas *****

lapply(Tx, function (x) {
  ggplot(df_iot, aes_string(x)) +
    geom_histogram() +
    ggtitle(paste("Histograma da Temperatura - ", x))
})

lapply(RH_x, function (x) {
  ggplot(df_iot, aes_string(x)) +
    geom_histogram() +
    ggtitle(paste("Histograma da Umidade - ", x))
})

# Comentários sobre os histogramas abaixo:

# 1) Verifica-se que há uma concentração de valores baixos para a variável
# Appliances(target) e também valores extremos à direita.
# 2) A variável lights apresenta muitos valores zerados (indica não uso)
# 3) O histograma da variável Press_mm_hg tem formato assimétrico à esquerda
# 4) Os histogramas das variáveis Windspeed e Tdewpoint são assimétricos à direita

hist(df_iot$Appliances, main = "Histograma dos Eletrodomésticos - IoT")
hist(df_iot$lights, main = "Histograma das Luminárias")
hist(df_iot$Press_mm_hg, main = "Histograma da Unidade de Pressão")
hist(df_iot$Windspeed, main = "Histograma da Velocidade do Vento")
hist(df_iot$Visibility, main = "Histograma da Visibilidade")
hist(df_iot$Tdewpoint, main = "Histograma do Ponto de Condensação da Água")
hist(df_iot$NSM, main = "Histograma de North Star Metric")
hist(df_iot$rv1, main = "Histograma de rv1")



# Plots por dia da semana (útil e fim de semana)

# Comentários sobre os plots abaixo:

# 1) O consumo nos dias não úteis é menor em relação ao consumo dos dias úteis,
# mas isso pode ser explicado pela maior quantidade de dias úteis.

# 2) No gráfico por dia da semana, observa-se uma tendência maior de consumo de
# energia nas sextas, sábados e domingos. 

ggplot(data = df_iot, aes(x = WeekStatus, y = Appliances, fill = WeekStatus)) +
  geom_bar(stat = 'identity') +
  ggtitle("Appliances por Dias Úteis/Não Úteis")


ggplot(data = df_iot, aes(x = Day_of_week, y = Appliances, fill = Day_of_week)) +
  geom_bar(stat = 'identity') +
  ggtitle("Appliances por Dia da Semana")



#***** Correlação *****

# Append das Variáveis de temperatura e umidade para gerar a correlação
Vars_num <- append(Tx, RH_x)

c_pearson <- cor(df_iot[Vars_num])
View(c_pearson)

# Matriz de correlação das Medidas de Temperaturas e Umidades
corrplot(c_pearson,
         method = 'color',
         cl.pos = "b",
         type = "lower",
         addgrid.col = "white",
         addCoef.col = "white",   # insere os coeficientes de correlação de Pearson
         tl.col = "black",   # muda a cor do nome das variáveis
         tl.cex = 0.7,   # tamanho do nome das variáveis
         number.cex = 0.7,
         cl.cex = 0.7
)

# Comentários sobre a correlação entre as medidas de temperatura e umidade: 
#
# 1) Verifica-se que a maioria das variáveis de temperatura estão fortemente
# correlacionadas entre si, com exceção dos pares (T8, T6), (T8, T_out) e (T8, T2)
# que apresentam uma correlação positiva mais fraca.

# 2) Observa-se que as variáveis RH_6 e RH_out estão correlacionadas negativamente 
# com as variáveis de temperatura. As demais variáveis de umidade apresentam-se
# fracamente correlacionadas com as variáveis de temperatura.

# 3) Verifica-se, também, que as variáveis de umidade estão bastante correlacionadas
# entre si, com exceção das variáveis RH_5, RH_6 e RH_out. Sendo que RH_out só apresenta
# uma correlação mais forte com RH_6 e RH_2.


# Verificando a correlação da variável target com todas as variáveis numéricas
corr_varnum <- cor(df_iot[ , 2:30])
View(corr_varnum)

# Matriz de correlação dos dados
corrplot(corr_varnum,
         method = 'color',
         cl.pos = "b",
         type = "lower",
         addgrid.col = "white",
         addCoef.col = "white",   # insere os coeficientes de correlação de Pearson
         tl.col = "black",   # muda a cor do nome das variáveis
         tl.cex = 0.7,   # tamanho do nome das variáveis
         number.cex = 0.7,
         cl.cex = 0.7
)

# Comentário: 
# Exceto a variável NSM que apresenta uma correlação positiva pequena com a 
# variável target (Appliances), NÃO se observa correlação significativa entre
# a variável target (Appliances) e as demais variáveis.



#***********************************************
#********* Pré-Processamento dos Dados *********
#***********************************************

str(df_iot)

# Vetor com algumas variáveis para averiguação
verif_cols <- c("Appliances", "lights", "NSM")

# Loop para verificar a quantidade de valores únicos de algumas variáveis
for (i in verif_cols) {
  print(paste(i,":" ,length(unique(df_iot[[i]]))))
}

# Convertendo o tipo da variável lights
# df_iot$lights <- as.factor(df_iot$lights)



#***** Features Engineering *****

# Formatando a variável date
df_iot$date <- as.POSIXct(strptime(
  paste(df_iot$date, " ", ":00:00", sep = ""), 
  "%Y-%m-%d %H:%M:%S"))

# Criando a variável hora dos registros de dados de energia extraídos da
# variável date
df_iot$hora <- as.integer(format(df_iot$date, "%H"))

# Verificando a distribuição da nova variável hora
table(df_iot$hora)

# Scatterplot
# Comentário: Observa-se alguns horários de pico de consumo de energia.
# Verifica-se que o período de maior uso de energia é de 7h até 20h. 
ggplot(data = df_iot, aes(x = hora, y = Appliances)) +
  geom_point() +
  ggtitle("Scatterplot - Appliances x Hora")


# Excluindo as colunas date e rv2 (valores idênticos a coluna rv1)
df_iot$date <- NULL
df_iot$rv2 <- NULL

str(df_iot)


# Aplicando One-hot Encoding nas variáveis categóricas (WeekStatus e Day_of_week)

# Criando a função para converter as variáveis categóricas em variáveis dummies
dmy <- dummyVars("~.", data = df_iot)

# Criando novos dataframes aplicando a função aos datasets de treino e teste
df_iot_dummies <- data.frame(predict(dmy, newdata = df_iot))

# Visualizando
View(df_iot_dummies)


# Criando um vetor com os nomes das variáveis numéricas
numeric.vars<- names(df_iot_dummies)[3:28]
numeric.vars

# Função para Normalização
scale.features <- function(df, variables){
  for(variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}

# Normalizando as variáveis
df_iot_scaled <- scale.features(df_iot_dummies, numeric.vars)

# Visualizando os dados normalizados
View(df_iot_scaled)



#***** Feature Selection *****

# Aplica randomForest para gerar um plot de importância das variáveis

modelo <- randomForest(Appliances ~ .,
                       data = df_iot_scaled,
                       ntree =100,
                       nodesize = 10,
                       importance = TRUE)

importancia <- varImpPlot(modelo, scale = FALSE)
View(importancia)


# Dataframe com a seleção das variáveis mais relevantes

df_scaled_selected <- df_iot_scaled %>%
  select(
    contains("Appliances"),
    contains("NSM"),
    contains("hora"),
    contains("T3"),
    contains("RH_3"),
    contains("RH_1"),
    contains("T8"),
    contains("RH_2")
  )



# ***** Removendo Valores Discrepantes da Variável Target *****

# A variável target (Appliances) tem muitos valores discrepantes. A performance 
# dos modelos preditivos é ruim quando a variável target tem muitos outliers.

summary(df_scaled_selected$Appliances)


# Calculando o percentil 90 para a variável target
q <- quantile(df_scaled_selected$Appliances,  probs = 0.90) 
q

# Eliminando os registros dos valores da variável target acima do percentil 90
df_scaled_select <- dplyr::filter(df_scaled_selected, Appliances <= q)

dim(df_scaled_select)



#***********************************************
# Divisão dos Dados em Treino, Validação e Teste
#***********************************************


# ***** Divisão dos Dados em Treino, Validação e Teste (sem Valores Discrepantes) *****

set.seed(182)

# Função do Caret para divisao dos dados
split <- createDataPartition(y = df_scaled_select$Appliances, p = 0.8, list = FALSE)

# Criando dados de treino e de teste
df_treino_scaled <- df_scaled_select[split,]
df_teste_scaled <- df_scaled_select[-split,]


dim(df_treino_scaled)
dim(df_teste_scaled)


# Fazendo uma nova divisão com os dados de treino para a validação 
set.seed(182)

split2 <- createDataPartition(y = df_treino_scaled$Appliances, p = 0.8, list = FALSE)

# Criando dados de treino e de teste
df_treino_scaled_select <- df_treino_scaled[split2,]
df_valida_scaled_select <- df_treino_scaled[-split2,]


dim(df_treino_scaled_select)
dim(df_valida_scaled_select)


# Removendo datasets para liberar a memória
rm(df_iot_dummies)
rm(df_scaled_selected)
rm(df_scaled_select)
rm(df_treino_scaled)




#***********************************************
#****** Construção dos Modelos Preditivos ******
#***********************************************


# *** Modelo Inicial de Regressão Linear Múltipla ***

# Primeiro verificaremos o comportamento do modelo de Regressão Linear

modelo_lm <- lm(Appliances ~ ., data = df_treino_scaled_select)

# Resumo do modelo 
summary(modelo_lm)


# Comentário:
# O coeficiente de determinação (R2) do modelo de regressão linear baixo (R2 
# ajustado aproximadamente 0.24) indica que o modelo não se ajusta aos dados. 



# Treinando o modelo de regressão com cross-validation 

controle <- trainControl(method = "cv", number = 100)

modelo_lm_control <- train(Appliances ~ .,
                           data = df_treino_scaled_select, 
                           method = "lm", 
                           trControl = controle, 
                           metric = "Rsquared")

# Print
print(modelo_lm_control)

# Comentário: O modelo treinado com cross-validation não melhorou o coeficiente
# de determinação (R2 ajustado de 0.24).


# Coletando os residuos
residuals <- resid(modelo_lm)

# Plot dos resíduos
# Comentário: Verifica-se no gráfico que os resíduos referentes aos valores
# extremos da variável target (Appliances) são elevados.
plot(residuals)
plot(df_treino_scaled_select$Appliances, residuals)

# Histograma dos resíduos
hist(residuals, main = "Histograma dos Resíduos do Modelo de Regressão Linear")

# QQ Plot
# gráfico dos quantis de resíduos (eixo das ordenadas) versus os quantis da
# normal padrão (eixo das abscissas)
qqnorm(residuals)
qqline(residuals, col = 'red')


# Comentários:

# Verifica-se que o histograma dos resíduos tem um formato assimétrico à direita.

# O gráfico de quantis mostra que os resíduos NÃO são normalmente distribuídos,
# o QQ Plot apresenta uma distorção à direita.




# ***** 1. Modelo Linear Generalizado (GLM) *****

# Como a variável target é assimétrica criaremos o modelo GLM com a distribuição
# gama.

modelo_glm <- glm(Appliances ~ .,
                  family = Gamma(link = "identity"),
                  data = df_treino_scaled_select)

# Resumo do modelo GLM
summary(modelo_glm)

# Previsão com dados de validação
previsoes_glm <- predict(modelo_glm, newdata = df_valida_scaled_select)

# Calcula a métrica mae (mean absolute error)
mae_glm <- mae(df_valida_scaled_select$Appliances, previsoes_glm)
mae_glm

# Calculando a métrica rmse (root mean squared error)
rmse_glm <- rmse(df_valida_scaled_select$Appliances, previsoes_glm)
rmse_glm



# Treinando o modelo GLM com cross-validation 
control_glm <- trainControl(method = "cv", number = 100)

modelo_glm_control <- train(Appliances ~ .,
                            data = df_treino_scaled_select, 
                            method = "glm",
                            metric = "RMSE",
                            trControl = control_glm)

# Print do Modelo GLM com a aplicação de cross validation
print(modelo_glm_control)

summary(modelo_glm_control)


# Comentários sobre o modelo GLM:
# 1) O coeficiente de determinação (R2) do modelo GLM também foi baixo de 
# aproximadamente 0.25, indicando que não houve melhora com o modelo GLM. 

# As métricas MAE (mean absolute error) de 19 e o RMSE (root mean squared error)
# de 26 obtidas com o modelo GLM foram iguais as do modelo de regressão linear.

# O treinamento com cross-valiation não apresentou melhora, o AIC (Akaike
# Information Criteria) de 106902 foi um pouco maior do que o modelo treinado
# sem cross-validation (AIC: 103316).



#***** 2. Modelo SVM (Support Vector Machine) *****

# Primeira Versão do Modelo SVM - Versão Padrão com Kernel Radial (RBF)
# O algoritmo escolhe o tipo de SVM de acordo com o tipo de dado da variável target

modelo_svm1 <- svm(Appliances ~ ., 
                   data = df_treino_scaled_select,
                   na.action = na.omit, scale = FALSE)

summary(modelo_svm1)

# Previsões com os dados de validação
previsoes_svm1 <- predict(modelo_svm1, newdata = df_valida_scaled_select)

# Calcula a métrica mae (mean absolute error)
mae_svm1 <- mae(df_valida_scaled_select$Appliances, previsoes_svm1)
mae_svm1

# Calculando a métrica rmse (root mean squared error)
rmse_svm1 <- rmse(df_valida_scaled_select$Appliances, previsoes_svm1)
rmse_svm1



# Segunda Versão do Modelo SVM - Kernel Linear

modelo_svm2 <- svm(Appliances ~ ., 
                   data = df_treino_scaled_select,
                   kernel = 'linear',
                   na.action = na.omit, scale = FALSE)

summary(modelo_svm2)

# Previsões com os dados de validação
previsoes_svm2 <- predict(modelo_svm2, newdata = df_valida_scaled_select)

# Calcula a métrica mae (mean absolute error)
mae_svm2 <- mae(df_valida_scaled_select$Appliances, previsoes_svm2)
mae_svm2

# Calculando a métrica rmse (root mean squared error)
rmse_svm2 <- rmse(df_valida_scaled_select$Appliances, previsoes_svm2)
rmse_svm2



# Terceira Versão do Modelo SVM - Kernel Polinomial

# Fazendo uma pesquisa em grades para ajustar o parâmetro (degree) com kernel
# polinomial. Não consideraremos grau polinomial de ordem superior a 4, para
# evitar um ajuste excessivo.

set.seed(182)

modelo_svm3_grid <- tune(svm,
                         Appliances ~ ., 
                         data = df_treino_scaled_select,
                         kernel = 'polynomial',
                         ranges = list(degree = c(2, 3, 4)))

# Parâmetros do melhor modelo
modelo_svm3_grid$best.parameters

# Melhor modelo
modelo_svm3 <- modelo_svm3_grid$best.model
summary(modelo_svm3)

# Previsões com dados de validação
previsoes_svm3 <- predict(modelo_svm3, newdata = df_valida_scaled_select)

# Calculando a métrica mae (mean absolute error)
mae_svm3 <- mae(df_valida_scaled_select$Appliances, previsoes_svm3)
mae_svm3

# Calculando a métrica rmse (root mean squared error)
rmse_svm3 <- rmse(df_valida_scaled_select$Appliances, previsoes_svm3)
rmse_svm3


# Comentários sobre os modelos SVM:
# O modelo SVM com kernel RBF apresentou o melhor desempenho dentre os modelos
# SVM criados, as métricas MAE e RMSE do modelo SVM com Kernel RBF foram menores.   



# ***** 3. Modelo XGBOOST (eXtreme Gradient Boosting) ***** 

str(df_treino_scaled_select)

# Transformando o dataframe em uma matriz
df_treino_scaled_data <- as.matrix(df_treino_scaled_select[ , 2:8])
class(df_treino_scaled_data)

modelo_xgb <- xgboost(data = df_treino_scaled_data,
                     label = df_treino_scaled_select$Appliances,
                     max.depth = 2,
                     eta = 1,   # controla a learning rate (evita overfitting)
                     nthread = 2, 
                     nround = 10, # número de passadas nos dados de treino
                     objective = "reg:linear",
                     eval_metric = "rmse")

# Imprimindo o modelo
print(modelo_xgb)

# Transformando o dataframe de validação em uma matriz de variáveis preditoras
df_valida_scaled_data <- as.matrix(df_valida_scaled_select[ , 2:8])
class(df_valida_scaled_data)

# Previsões com a matriz de dados de validação
previsoes_xgb <- predict(modelo_xgb, df_valida_scaled_data)

# Calcula a métrica mae (mean absolute error)
mae_xgb <- mae(df_valida_scaled_select$Appliances, previsoes_xgb)
mae_xgb

# Calcula a métrica rmse (root mean squared error)
rmse_xgb <- rmse(df_valida_scaled_select$Appliances, previsoes_xgb)
rmse_xgb


# Criando a Matriz de Importância de Atributos
importance_matrix <- xgb.importance(model = modelo_xgb)
print(importance_matrix)

# Plot
xgb.plot.importance(importance_matrix = importance_matrix)



#***********************************************
# *********** Avaliação dos Modelos ************
#***********************************************

# Vetores com as métricas de avaliação dos modelos
vetor_glm <- c(round(mae_glm, 4), round(rmse_glm, 4))
vetor_svm1 <- c(round(mae_svm1, 4), round(rmse_svm1, 4))
vetor_svm2 <- c(round(mae_svm2, 4), round(rmse_svm2, 4))
vetor_svm3 <- c(round(mae_svm3, 4), round(rmse_svm3, 4))
vetor_xgb <- c(round(mae_xgb, 4), round(rmse_xgb, 4))

# Concatenando as métricas
compara_modelos <- rbind(vetor_glm, vetor_svm1, vetor_svm2, vetor_svm3, vetor_xgb)
colnames(compara_modelos) <- c("MAE", "RMSE")

class(compara_modelos)

# Transformando em dataframe
compara_modelos <- as.data.frame(compara_modelos)

vetor_metricas <- c("GLM", "SVM Kernel RBF", "SVM Kernel Linear", "SVM Kernel Polinomial", "XGBOOST")
compara_modelos$Modelo <- vetor_metricas

# Visualizando o dataframe de comparação dos modelos
View(compara_modelos)

# Plot MAE
ggplot(compara_modelos, aes(x = Modelo, y = MAE, fill = Modelo)) +
  geom_bar(stat = 'identity') +
  ggtitle("Comparação do MAE dos Modelos")

# Plot RMSE
ggplot(compara_modelos, aes(x = Modelo, y = RMSE, fill = Modelo)) +
  geom_bar(stat = 'identity') +
  ggtitle("Comparação do RMSE dos Modelos")


# Comentários sobre os modelos preditivos:

# O modelo SVM com kernel RBF apresentou o melhor desempenho em relação as
# métricas MAE (15.8128) e RMSE (24.2412), o modelo XGBoost foi o segundo
# menor.

# O RMSE (raiz do erro médio quadrático) dos modelos foi maior do que o MAE
# (erro médio absoluto), isso pode ser devido a métrica RMSE penalizar quando
# os valores previstos apresentam valores distantes do real. 

# Provavelmente se aumentasse o parâmetro nround no treinamento do XGBoost
# obteríamos métricas melhores, mas tendo em vista o custo/benefício de
# recursos computacionais ficaremos com o modelo SVM.



#***********************************************
# ****** Otimização do Modelo Selecionado ******
#***********************************************

# ***** Tuning do Modelo SVM - Kernel RBF *****

# Fazendo uma pesquisa em grade para o ajuste de parâmetros, sem considerar
# valores muito baixo para gamma, para evitar um excesso de ajuste.
# O custo não superior a 2 para que valores discrepantes não afetem os
# limites de decisão e, portanto, levem ao ajuste excessivo (overfitting).

set.seed(182)

modelo_svm_grid <- tune(svm,
                        Appliances ~ ., 
                        data = df_treino_scaled_select,
                        kernel = 'radial',
                        ranges = list(cost = c( 1, 2),
                                      gamma = c(0.01, 0.1, 1)))

# Parâmetros do melhor modelo
modelo_svm_grid$best.parameters
summary(modelo_svm_grid)

# Melhor modelo
best_svm <- modelo_svm_grid$best.model


# Previsões com dados de validação
previsoes_best_svm <- predict(best_svm, newdata = df_valida_scaled_select)

# Calculando a métrica mae (mean absolute error)
mae_best_svm <- mae(df_valida_scaled_select$Appliances, previsoes_best_svm)
mae_best_svm

# Calculando a métrica rmse (root mean squared error)
rmse_best_svm <- rmse(df_valida_scaled_select$Appliances, previsoes_best_svm)
rmse_best_svm



#***********************************************
# ***** Conclusão: Modelo Preditivo Final ******
#***********************************************

# O modelo SVM com Kernel RBF com ajuste de hiperparâmetros melhorou a
# performance, tanto o erro médio absoluto (MAE) quanto a raiz do erro
# médio quadrático (RMSE) deste modelo foram menores:

# best parameters:
# cost gamma
#   2     1

# MAE do SVM ajustado: 13.20821 
# RMSE do SVM ajustado: 20.78178



#***********************************************
# ********** Previsão com Novos Dados ********** 
#***********************************************

# Fazendo previsão com os dados separados para teste

previsoes_novos_dados <- predict(best_svm, newdata = df_teste_scaled)

# Calculando a métrica mae (mean absolute error) com dados de teste
mae_novos_dados <- mae(df_teste_scaled$Appliances, previsoes_novos_dados)
mae_novos_dados

# Calculando a métrica rmse (root mean squared error) com dados de teste
rmse_novos_dados <- rmse(df_teste_scaled$Appliances, previsoes_novos_dados)
rmse_novos_dados


# Resíduos
df_erro <- data.frame(previsoes_novos_dados - df_teste_scaled$Appliances)
names(df_erro) <- c("residuos")

# Resumo das medidas de tendência central dos resíduos
summary(df_erro$residuos)

# Plotando os resíduos
ggplot(df_erro, aes(x = residuos)) +
  geom_histogram(binwidth = 1, fill = "white", color = "black") +
  ggtitle("Resíduos do Modelo SVM com kernel RBF e Otimização de Parâmetros")



# ************ Considerações Finais ************
#
# As métricas para avaliação da previsão com novos dados para teste ficaram
# próximas do modelo treinado:

# MAE do SVM ajustado com novos dados: 12.73054 
# RMSE do SVM ajustado com novos dados: 19.66485

