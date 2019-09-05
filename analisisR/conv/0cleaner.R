setwd('C:/Users/Lorenzo/Documents/TFM/CODES/7.- METRICS/METRICS CONV/metrics/')
require(data.table) 
library(tidyverse)

# IMPORT DATA:
folder <- "." #PATH
list.all <- list.files(path=folder, pattern=".csv")
metrics.all = rbindlist(lapply(list.all, fread), fill = T)

#Renombro:
name <- c('net', 'total_time', 'epoch_time', 'step_time', 'mse', 'inference_time', 'parameters', 'name', 'batch_size', 'learning_rate', 'opt', 'model', 'epochs')
names(metrics.950M) <- name 
names(metrics.980) <- name 
names(metrics.1080) <- name 

#LIMPIO STRINGS Y AÑADO device_name:
metrics_950M<- metrics.950M %>%
  mutate(device = str_sub(net, start = -3, -1), #Cogemos solo desde la posición 16
         topology = substr(net, 1, 15),
         net = str_sub(net, 17,-5)) %>%
  group_by(device) %>%
  mutate(device_name = ifelse(device == 'GPU','GTX950M', 'IntelCorei7'))

metrics_980<- metrics.980 %>%
  mutate(device = str_sub(net, start = -3, -1), #Cogemos solo desde la posición 16
         topology = substr(net, 1, 15),
         net = str_sub(net, 17,-5)) %>%
  group_by(device) %>%
  mutate(device_name = ifelse(device == 'GPU','GTX980', 'IntelXeonE5'))

metrics_1080<- metrics.1080 %>%
  filter(str_sub(net, start = -3, -1) == 'GPU') %>%
  mutate(device = str_sub(net, start = -3, -1), #Cogemos solo desde la posición 16
         topology = substr(net, 1, 15),
         net = str_sub(net, 17,-5),
         device_name = 'GTX1080') 


#MAKE TOTAL
metrics_total <- metrics_950M %>%
  bind_rows(metrics_1080, metrics_980) %>%
  mutate(net = str_sub(net, start=2, -2))

#CLEAN TOTAL
metrics_1 <-  metrics_total %>%
  filter(str_count(net,' ')== 0) %>%
  mutate(
         hidden_layers = 1,
         nn_layer_1 = net,
         nn_layer_2 = '0',
         nn_layer_3 = '0')
#STRINGR
#sacar las palabras separadas por ' ' y seleccionar de la lista
metrics_2 <-  metrics_total %>%
  filter(str_count(net,' ')== 1) %>%#encontrar veces hay espacio 
  group_by(name) %>%
         mutate(
         hidden_layers = 2,
         nn_layer_1 = str_extract_all(net, boundary("word"))[[1]][1],
         nn_layer_2 = str_extract_all(net, boundary("word"))[[1]][2],
         nn_layer_3 = '0')

metrics_3 <-  metrics_total %>%
  filter(str_count(net,boundary('word'))== 3) %>% #encontrar veces hay palabra
  group_by(name) %>%
  mutate(
    hidden_layers = 3,
    nn_layer_1 = str_extract_all(net, boundary("word"))[[1]][1],
    nn_layer_2 = str_extract_all(net, boundary("word"))[[1]][2],
    nn_layer_3 = str_extract_all(net, boundary("word"))[[1]][3])

metrics <- metrics_1 %>%
  bind_rows(metrics_2, metrics_3)

write_csv(metrics.all, '../metrics_total.csv',col_names = T)
