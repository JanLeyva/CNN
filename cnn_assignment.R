if(!require("knitr")) install.packages("knitr")
if(!require("keras")) install.packages("keras")
if(!require("kerasR")) install.packages("kerasR")
if(!require("tfruns")) install.packages("tfruns")

library("knitr")
library("keras")
library(kerasR)
library(tfruns)


base_dir<-"~/KOLMOGOROV/Assignment_CNN/malaria/malaria"
# train directories
train_dir<-file.path(base_dir,"train",fsep ="/")
train_dir_infected<-file.path(train_dir,"infected",fsep = "/")
train_dir_uninfected<-file.path(train_dir,"uninfected",fsep = "/")

# vaidation directories
validation_dir<-file.path(base_dir,"validation",fsep ="/")
validation_dir_infected<-file.path(validation_dir,"infected",fsep = "/")
validation_dir_uninfected<-file.path(validation_dir,"uninfected",fsep = "/")

# test directories
test_dir<-file.path(base_dir,"test",fsep ="/")
test_dir_infected<-file.path(test_dir,"infected",fsep = "/")
test_dir_uninfected<-file.path(test_dir,"uninfected",fsep = "/")


# image_data_generator Generate batches of image data with real-time data augmentation. The data will be looped over (in batches).
train_datagen <- image_data_generator(rescale = 1/255) #
validation_datagen <- image_data_generator(rescale = 1/255) #
#flow_images_from_directory Generates batches of data from images in a directory (with optional augmented/normalized data)
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(64, 64),
  batch_size = 20,
  class_mode = "binary"
)
validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(64, 64),
  batch_size = 20,
  class_mode = "binary"
)
batch <- generator_next(train_generator)
str(batch) 


## CNN definition -------------------------------------------------
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(64, 64, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


# Define the parameters
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy")
)


history <- model %>% fit(
  train_generator,
  steps_per_epoch = 100, #100
  epochs = 5, # 20
  validation_data = validation_generator,
  validation_steps = 50 #50
)

# Define FLAGS ---------------------------------------------------------
FLAGS <- flags(
  flag_string("hl1", "batch size"))


# fit model -------------------------------------------------

history <- model %>% fit(
  train_generator, train_generator$labels,
  epochs = 10, # 10
  batch_size = FLAGS$hl1
)
# evaluate
model %>% evaluate(validation_generator, validation_generator$labels)



y_pred <-model %>% 
  predict(validation_generator)

y_pred_ <- c()
for(i in 1:length(y_pred)){
  ifelse(y_pred[i] > 0.50001, y_pred_[i] <- 1, y_pred_[i] <- 0)
}

