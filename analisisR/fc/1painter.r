setwd('C:/Users/Lorenzo/Documents/TFM/CODES/7.- METRICS/METRICS FULLY-CONNECTED/')
library (tidyverse)

metrics <- read_csv('metrics/metrics_total.csv')

metrics <- metrics %>%
  mutate(net=as.character(net), inference_time = inference_time*100)

#VISUALIZATION
#Comparación de todos los dispositivos:
por_entrenamiento <-  metrics %>%
  mutate(net = fct_reorder(net, total_time)) %>%
  ggplot(aes(x=net, y=total_time, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de entrenamiento por red') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
  #ggsave('entrenamiento.jpeg')


#####################################################################
#Comparación de entrenamiento por CPUs
entrenamiento_params <-  metrics %>%
  filter(batch_size == 1000, device == 'CPU', parameters < 1300000,
         net != '12800') %>%
  mutate(net = fct_reorder(net, parameters)) %>%
  ggplot(aes(x=net, y=total_time, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de entrenamiento por red',
        subtitle = 'Redes ordenadas por su número de parámetros') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('entrenamiento_cpus.jpg')


entrenamiento_params <-  metrics %>%
  mutate(net = fct_reorder(net, parameters)) %>%
  filter(batch_size == 1000, device_name =='GTX1080'| device_name == 'IntelXeonE5',
         parameters > 120000, parameters < 103885057) %>%
  ggplot(aes(x=net, y=total_time, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de entrenamiento por red',
        subtitle = 'Redes ordenadas por su número de parámetros') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('E5vs1080_2.jpg')

entrenamiento_params <-  metrics %>%
  mutate(net = fct_reorder(net, parameters)) %>%
  filter(batch_size == 1000, device_name =='GTX1080'| device_name == 'IntelXeonE5',
         parameters < 120000) %>%
  group_by(device_name)%>%
  ggplot(aes(x=net, y=total_time, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de entrenamiento por red',
        subtitle = 'Redes ordenadas por su número de parámetros') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('E5vs1080_1.jpg')

entrenamiento_params

####################################################################
#WIDE VS DEEP

entrenamiento_params <-  metrics %>%
  filter(batch_size == 1000, device_name == 'GTX1080',
         parameters > 166401, parameters < 20812801,
         epochs == 2) %>%
  mutate(net = fct_reorder(net, parameters)) %>%
  ggplot(aes(x=net, y=epoch_time, color = as.factor(hidden_layers))) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de entrenamiento por red',
        subtitle = 'Redes ordenadas por su número de parámetros',
        color = 'Capas ocultas') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('1080-widevsdeep.jpg')
####################################################################

#SUPERACIÓN DE LA GPU A CPU CON BATCH 100
gpu_supera_cpu <-  metrics %>%
  filter(batch_size == 100, parameters >332801, parameters < 5286401) %>%
  mutate(net = fct_reorder(net, parameters)) %>%
  ggplot(aes(x=net, y=total_time, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de entrenamiento por red',
        subtitle = 'Redes ordenadas por su número de parámetros') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#ggsave()
gpu_supera_cpu

#Podría coger las de una capa, i7 vs 950, luego las de dos y luego las de 3


###############################
#Comparación de las GPUs:
##############################
gpus <- metrics %>%
  filter(device_name %in% c('GTX1080', 'GTX980', 'GTX950M'))

gpus_entrenamiento <-  gpus %>%
  filter(batch_size == 1000) %>%
  mutate(net = fct_reorder(net, parameters)) %>%
  ggplot(aes(x=net, y=total_time, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de entrenamiento por red',
        subtitle = 'Redes ordenadas por su número de parámetros') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
  #ggsave()
gpus_entrenamiento
#######################DONEDONEDONEDONE###################################
por_entrenamiento_batch <-  metrics %>%
  ggplot(aes(x=net, y=batch_size, color = batch_size)) +
  geom_point(stat="identity") +
  labs( title = 'MSE en función del batch size',
        subtitle = 'En todos los dispositivos') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('entrenamiento_batch.jpeg')

#########################DONEDONEDONE###################################
por_entrenamiento_batch <-  metrics %>%
  filter(net == '12800') %>%
  ggplot(aes(x=batch_size, y=step_time, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de lote para cada dispositivo',
        subtitle = 'Red de una capa oculta de 12800 neuronas') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('entrenamiento_s2.jpeg')

por_entrenamiento_batch <-  metrics %>%
  filter(net == '50') %>%
  ggplot(aes(x=batch_size, y=step_time, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de lote para cada dispositivo',
        subtitle = 'Red de una capa oculta de 50 neuronas') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('entrenamiento_s1.jpeg')


por_entrenamiento_batch <-  metrics %>%
  filter(net == '12800') %>%
  ggplot(aes(x=batch_size, y=step_time, color = device_name)) +
  geom_line(stat="identity") +
  labs( title = 'Evolución del tiempo de lote según su tamaño',
        subtitle = 'Red de una capa oculta de 12800 neuronas') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('entrenamiento_s3.jpeg')



f_3 <- metrics %>%
  filter(net == 50 & device_name %in% c('IntelCorei7','GTX1080'))%>%
  ggplot(aes(x=batch_size, y=total_time, color= device_name)) +
  geom_line(stat="identity") +
  labs( title = 'Tiempo de entrenamiento vs batch ',
        subtitle = 'Red de una capa oculta de 50 neuronas') +
  ggsave('total_time vs batch_size.png')


f_3 <- metrics %>%
  filter(net == '6400' & device_name %in% c('IntelCorei7','GTX1080'))%>%
  ggplot(aes(x=batch_size, y=total_time, color= device_name)) +
  geom_line(stat="identity") +
  labs( title = 'Tiempo de entrenamiento vs batch ',
        subtitle = 'Red de una capa oculta de 12800 neuronas') 

f_3

f_1 <- metrics %>%
  filter(net == '6400 3200 1664' , device_name == 'IntelCorei7')

#################################################################
# Reorder following the value of another column:
por_inferencia <-  metrics %>%
  mutate(net = fct_reorder(net, inference_time)) %>%
  ggplot(aes(x=net, y=inference_time, color = batch_size)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de inferencia por red') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
  #ggsave('inferencia.jpeg')
por_inferencia
#A XEONE5 le cuesta con el paso de 12800 a 6400

#####################################################33
#RELACIÓN TIEMPO TOTAL - TIEMPO EPOCH
t_epoch <- metrics %>%
  filter(device_name %in% c('GTX980', 'IntelXeonE5', 'GTX1080')) %>%
  mutate(rel = total_time/epoch_time) %>%
  group_by(device_name, epochs) %>%
  summarize(
    relation = mean(rel)
    )%>%
  arrange(desc(relation)) %>%
  write_csv('t_totaltime-epochtime.csv')

t_step <- metrics %>%
  mutate(train_steps = ceiling(total_time/step_time)) %>%
  group_by(device_name) %>%
  summarize(
    relation = mean((train_steps*step_time)/total_time)
  )%>%
  arrange(desc(relation)) %>%
  write_csv('t_totaltime-epoch.csv')

step_time_12800 <- metrics %>%
  filter (net == 12800, epochs == 1)%>%
  group_by(device_name) %>%
  summarise(avg_step_time = mean(step_time))%>%
  arrange(desc(avg_step_time))

step_time_all_nets <- metrics %>%
#  filter (batch_size==1000)%>%
  group_by(net, device_name, total_time) %>%
  summarise(avg_step_time = mean(step_time))%>%
  arrange(desc(avg_step_time))

step_time_50 <- metrics %>%
  filter (net == 50, epochs == 1 )%>%
  group_by(device_name) %>%
  summarise(avg_step_time = mean(step_time))%>%
  arrange(desc(avg_step_time))

###################################################



########################
#INFERENCIA
######################
#El conjunto total para inferencia ha sido de 290000 observaciones
#Calcular el número de pasos realizados para cada batch:

metrics <- metrics %>%
  mutate(inf_steps = ceiling(290000/batch_size)) %>%
  mutate(sample_time = inference_time/inf_steps)

a <- metrics %>%
  filter (batch_size %in% c(500, 1000), net == '6400 3200 1664')

ggplot(data = a) +
  geom_col(mapping = aes(x = device_name, y = sample_time, fill = device),
           position = position_stack(reverse = TRUE)) +
  facet_wrap(~ batch_size)+
  coord_flip() +
  labs( title = 'Tiempo de inferencia por dispositivo',
        subtitle = 'Conjuntos de muestras',
        x = ' ', y = ' ', fill = ' ') + 
  theme(legend.position = "top") + 
  ggsave('inferencia_6400 3200 1664.jpeg')
#La mayoría están por debajo de 0.05.
#0.5 y 0.17 son de las dos grandes CPU
#A partir de [400, 200] la inferencia ya es más rápida en GPU.

