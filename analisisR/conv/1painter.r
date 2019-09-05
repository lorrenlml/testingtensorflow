setwd('C:/Users/Lorenzo/Documents/TFM/CODES/7.- METRICS/METRICS CONV/')
library (tidyverse)

metrics <- read_csv('metrics_total.csv')

metrics2 <- metrics %>%
  mutate(inf_steps = ceiling(45322/batch_size)) %>%
  mutate(sample_time = inference_time/inf_steps)

#VISUALIZATION

#ENTRENAMIENTO
a <- metrics2 %>%
  filter ( batch_size == 2, device == 'GPU')

ggplot(data = a) +
geom_col(mapping = aes(x = device_name, y = inference_time, fill = device),
  position = position_stack(reverse = TRUE)) +
coord_flip() +
labs( title = 'Tiempo de entrenamiento por dispositivo',
      x = ' ', y = ' ', fill = ' ') + 
theme(legend.position = "top") #+ 
#ggsave('train_conv.jpeg')



ggplot(data = a) +
  geom_col(mapping = aes(x = device_name, y = total_time, fill = device_name),
           position = position_stack(reverse = TRUE)) +
  coord_flip() +
  labs( title = 'Tiempo de entrenamiento en GPUs',
        fill ='', x = ' ' , y = ' ') + 
  theme(legend.position = "top") + 
  ggsave('traingpu_ conv.jpeg')#Comparación de todos los dispositivos:


########################
#INFERENCIA
######################
#El conjunto total para inferencia ha sido de 45322 imágenes
#Calcular el número de pasos realizados para cada batch:

a <- metrics2 %>%
  filter (batch_size %in% c(512, 1024))

ggplot(data = a) +
  geom_col(mapping = aes(x = device_name, y = sample_time, fill = device),
           position = position_stack(reverse = TRUE)) +
  facet_wrap(~ batch_size)+
  coord_flip() +
  labs( title = 'Tiempo de inferencia por dispositivo',
        subtitle = 'Conjuntos de muestras',
        x = ' ', y = ' ', fill = ' ') + 
  theme(legend.position = "top") + 
  ggsave('inferencia_512_1024_muetras.jpeg')

##############
### 1 #######
#############

#GRID DE BATCH PARA DISPOSITIVOS GENERAL:
# Da una breve exposición de la diferencia entre CPU y GPU.
# Permite ver como con batches muy pequeños el tiempo ya se dispara en las CPUs
# proporcionalmente. en las GPUS tambien.
por_entrenamiento_grid <-  metrics %>%
  filter(batch_size < 1000, batch_size > 2) %>%
  mutate(net = fct_reorder(device_name, total_time)) %>%
  ggplot(aes(x=device_name, y=total_time, color = device)) +
  geom_point(stat="identity") +
  facet_wrap(~ batch_size) +
  labs( title = 'Tiempo de entrenamiento por dispositivo') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('entrenamiento.jpeg')

por_entrenamiento_grid <-  metrics %>%
  filter(batch_size < 1000, batch_size > 2, device == 'GPU') %>%
  mutate(net = fct_reorder(device_name, total_time)) %>%
  ggplot(aes(x=device_name, y=total_time, color = device_name)) +
  geom_point(stat="identity") +
  facet_wrap(~ batch_size) +
  labs( title = 'Tiempo de entrenamiento por dispositivo') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('entrenamiento_gpus.jpeg')



##Hacer un grip igual para las GPUs y ver la diferencia de magnitud 
# del eje de tiempo con esta primera. Sacar la relación que hay:
# Que sea de 300 veces más


por_entrenamiento <-  metrics %>%
  filter(batch_size < 1000, total_time < 2000)
  mutate(net = fct_reorder(device_name, total_time)) %>%
  ggplot(aes(x=device_name, y=total_time, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de entrenamiento por dispositivo') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('entrenamiento_grid.jpeg')

##############
# 2  #########
#############
#Evolución de la implicación del batch a medida que disminuye
por_entrenamiento_batch <-  metrics %>%
  filter(batch_size < 1000, total_time < 2000) %>%
  ggplot(aes(x=batch_size, y=total_time, color = device_name)) +
  geom_line(stat="identity") +
  labs( title = 'Evolución del tiempo en función del batch size',
        subtitle = 'En todos los dispositivos') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggsave('entrenamiento_batch.jpeg')

  
  
  
entrenamiento <- metrics %>%
  #group_by(batch_size) %>%
  ggplot() +
  geom_histogram(mapping = aes(x = total_time)) 

entrenamiento

#Comparación de las GPUs:
gpus <- metrics %>%
  filter(device_name %in% c('GTX1080', 'GTX980', 'GTX950M'))

some <- metrics %>%
  filter(device_name != 'IntelCorei7')

gpus_entrenamiento <-  gpus %>%
  group_by(batch_size) %>%
  mutate(net = fct_reorder(net, total_time)) %>%
  ggplot(aes(x=net, y=total_time, shape = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'Tiempo de entrenamiento por red') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
  #ggsave()
gpus_entrenamiento

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


por_mse <-  metrics %>%
  mutate(net = fct_reorder(net, total_time)) %>%
  ggplot(aes(x=net, y=mse, color = device_name)) +
  geom_point(stat="identity") +
  labs( title = 'mse por red ordenadas por total_time') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  #ggsave('mse.jpeg')

#La mayoría están por debajo de 0.05.
#0.5 y 0.17 son de las dos grandes CPU
#A partir de [400, 200] la inferencia ya es más rápida en GPU.

ggplot(data = two_layers) +
  geom_point(mapping = aes(x = inference_time, y = total_time))

ggplot(data = two_layers) +
  geom_histogram(mapping = aes(x = mse))


##########################333
#Distrib
centralized <- list(0.15931232968966166,	0.15364981412206996,	14.67154455184936, 14.671544551849365,	9.288779973983765,	1028,	1,	1,	50,	50,	'GPU',"GTX1080&980x2", "central")
distributed <- list(0.158346847693125,	0.15364981412206996,	13.423649072647095	,13.423649072647095,	7.5725791454315186	,1024	,1	,1	,50,	50,	"GPU",	"GTX1080&980x2", 'mirrored')
distributed_1 <- list(0.15857402844862503,	0.15847114362916076,	11.255912780761719,	11.255912780761719	,7.0605628490448	,1024	,1	,1	,50,	50,	'GPU',	'GTX1080&980', 'mirrored')
distributed_2 <- list(0.15866280618039044,	0.15847114362916076,	12.042720556259155,	12.042720556259155,	7.245966672897339,	1024,	1,	1,	50,	50,	'GPU',	'GTX980x2', 'mirrored')

dist <- metrics %>%
  select(-c('loss_train', 'loss_inf')) %>%
  mutate(strategy = "No") %>%
  rbind(centralized, distributed, distributed_1,distributed_2) %>%
  filter(batch_size == 1024 | batch_size == 1028, device == 'GPU', strategy == 'mirrored')


ggplot(data = dist) +
  geom_col(mapping = aes(x = device_name, y = total_time, fill = device_name),
           position = position_stack(reverse = TRUE)) +
  coord_flip() +
  labs( title = 'Tiempo de entrenamiento ',
        subtitle = 'Estrategia Mirrored',
        x = ' ', y = ' ', fill = ' ') + 
  theme(legend.position = "top") + 
  ggsave('1111scen.jpeg')

