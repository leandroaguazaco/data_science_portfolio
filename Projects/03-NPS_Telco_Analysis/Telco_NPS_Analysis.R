# 0. Libraries ====

# Instalar librerías

# install.packages("tidyverse")
# install.packages("caret")
# install.packages("data.table")
# install.packages("janitor")
# install.packages("DataExplorer")
# install.packages("skimr")

# Cargar librerías

library(tidyverse)
library(caret)
library(data.table)
library(readxl)
library(janitor)
library(DataExplorer)
library(skimr)

# 1. Datos ====

# Lectura de datos, limpieza de nombres y conversión de variables categóricas
base_nps <- read_excel(path = "BD_NPS_PERSONAS_sample.xlsx", 
                       sheet = "BD_NPS_PERSONAS_sample", 
                       col_names = TRUE) %>% 
  column_to_rownames(., var = 'ID') %>% # Establecer ID
  clean_names() %>% # Limpiar nombres
  mutate(across(.cols = c(antigued, grupo_edad, gener, estado_civil, region,
                          uso_serv_cliente, data_usur, technology, band_7,
                          max_network_voice, max_network_voice, tipo_m1, max_rech),
                .fns = factor))
  

# 2. Análisis exploratorio de datos ====

# Tipos originales de datos
str(base_nps)

## 2.1 Valores faltantes ====

base_nps%>% 
  plot_missing(ggtheme = theme_bw(), 
               theme_config = theme(legend.position = "none"))

## 2.2 Distribución de variables numéricas ====

base_nps %>% # Histogramas
  select_if(is.numeric) %>% 
  plot_histogram(geom_histogram_args = list(color = "black"), 
                 title = "Distribution Quantitative Variables", 
                 ggtheme = theme_bw(), 
                 ncol = 2,
                 theme_config = theme(plot.title = element_text(hjust = 0.5, 
                                                                face = "bold"), 
                                      axis.title.x = element_text(face = "plain"),
                                      axis.title.y = element_text(face = "plain")))

## 2.3 Detección datos atípicos ====

base_nps %>% # Distribution: scale boxplot
  select_if(is.numeric) %>% 
  scale() %>%
  boxplot()

## 2.4 Normalidad multivariante ====

base_nps %>% # qqplot
  select_if(is.numeric) %>% 
  na.omit() %>% 
  plot_qq(ggtheme = theme_bw(), 
          geom_qq_args = list(alpha = 0.3, 
                              color = "gray"), 
          title = "Test de Normalidad Univariante - qqplot",
          ncol = 4, 
          nrow = 2, 
          theme_config = theme(plot.title = element_text(hjust = 0.5)))

## 2.5 Correlación ====

base_nps %>% 
  select_if(is.numeric) %>% 
  na.omit() %>% 
  plot_correlation(cor_args = list(method = "spearman"), 
                   geom_text_args = list(color = "white", 
                                         label.size = 0.20),
                   title = "Correlation Heatmap",
                   ggtheme = theme_bw(), 
                   theme_config = theme(axis.title.x = element_blank(),
                                        axis.text.x = element_text(angle = 90, 
                                                                   vjust = 0.4),
                                        axis.title.y = element_blank(), 
                                        plot.title = element_text(face = "bold", 
                                                                  hjust = 0.5), 
                                        legend.position = "right", 
                                        legend.title = element_blank())) + 
  viridis::scale_fill_viridis(option = "D", 
                              alpha = 0.8)

## 2.6 Frecuencia variables categóricas ====

base_nps %>% 
  plot_bar(ggtheme = theme_bw(), # Bar plot       
           ncol = 4, 
           nrow = 3,) 

# 3 Datos de entrenamiento y testeo ====

set.seed(3456)
trainIndex <- createDataPartition(base_nps$target, 
                                  p = 0.8, 
                                  list = FALSE, 
                                  times = 1)

train_base_nps <- base_nps[trainIndex, ]
test_base_nps  <- base_nps[-trainIndex,]

# 4 Preprocesamiento de datos ====

## 4.1 One Hot Encoding ====

dummy_data <- dummyVars(formula = ~ ., 
                        data = train_base_nps, 
                        drop2nd = FALSE)

train_base_nps <- predict(dummy_data, 
                          newdata = train_base_nps) %>% 
  as.data.frame()

## 4.2 Predictores correlacionados ====

cor_matrix <- cor(train_base_nps2 %>% select_if(is.numeric) %>% 
                    select(-target), 
                    method = "spearman")

highlycor <- findCorrelation(cor_matrix, 
                             cutoff = 0.75)

train_base_nps3 <- train_base_nps2[ , -highlycor]

## 4.3 Normalización y transformación de datos ====

pp_no_nzv <- preProcess(train_base_nps3 %>% select(-target), 
                        method = c("corr", "center", "scale", "YeoJohnson", "nzv"))

train_base_nps3 = predict(pp_no_nzv, 
                    newdata = train_base_nps3 %>% select(-target))

base_nps_final <- merge(train_base_nps3, 
                        train_base_nps %>% select(target),
                        by = "row.names",
                        sort = FALSE,
                        all = TRUE) %>% 
  column_to_rownames("Row.names")



# 5 Modelo de regresión logística ====

## 5.1 Función base glm() ====
log_model <- glm(target ~ ., 
                 data = train_base_nps,
                 family = binomial(link='logit'))

summary(log_model)

## 5.2 Usando la librería caret ====

# Variables predictoras de entrenamiento
x_train = train_base_nps %>% 
  select(-target)

# Corregir nombres
names(x_train) <- make.names(names(x_train), unique = TRUE)

# Variable objetivo de entrenamiento
y_train <- train_base_nps$target %>%
  as.factor()


# Modelo
set.seed(20230704) # 2023/07/04
log_model_caret <- train(x = x_train, 
                         y = y_train, # Con datos codificados
                         method = "glm",
                         family = "binomial",
                         preProcess = c("corr", "center", "scale", "YeoJohnson", "nzv"), # Normalización y transformación
                         metric = "Accuracy", # "Accuracy" o "ROC".
                         trControl = trainControl(method = "cv", # Validación cruzada
                                                  number = 4, #4-fold cross-validation),
                                                  #summaryFunction = twoClassSummary 
                                                  #classProbs = TRUE,
                                                  #savePredictions = TRUE
                                                  )
                         # tuneGrid = logGrid
                         ) 

print(log_model_caret)
log_model_caret # Modelo final
log_model_caret$preProcess # Preprocesamiento

# 6. Evaluación del modelo ====

## 6.1  Codificación del dataset de testeo
test_base_nps <- predict(dummy_data, 
                         newdata = test_base_nps) %>% 
  as.data.frame()

# Predictores de testeo
x_test <- test_base_nps %>% 
  select(-target)
names(x_test) <- make.names(names(x_test), unique = TRUE)

# Objetivo de testeo
y_test <- test_base_nps$target %>%
  as.factor()

## 6.2 Predicción del dataset de testeo
pred_log_model_caret <- predict(object = log_model_caret, 
                                newdata = x_test, 
                                type = "raw")

## 6.3 Matriz de confusión
caret::confusionMatrix(pred_log_model_caret, 
                       y_test)

# Variables Importance 
plot(varImp(log_model_caret))

# 7. Statistical assumptions ====

par(mfrow = c(2, 2)) # Change the panel layout to 2 x 2
plot(log_model) # 1. Upper left and bottom left graph
