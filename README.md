# Convolutional Neural Network


## The problem

Malaria is a deadly, infectious mosquito-borne disease caused by Plasmodium parasites. These parasites are transmitted by the bites of infected female Anopheles mosquitoes. With regular manual diagnosis of blood smears, it is an intensive manual process requiring proper expertise in classifying and counting the parasitized and uninfected cells. Typically this may not scale well and might cause problems if we do not have the right expertise in specific regions around the world. We are lucky to have researchers at the Lister Hill National Center for Biomedical Communications (LHNCBC), part of National Library of Medicine (NLM) who have carefully collected and annotated this dataset of healthy and infected blood smear images. There is a balanced dataset of 13779 malaria and non-malaria (uninfected) cell images. The dataset consist of near thousand images downloaded from the official website that have been resized (64x64x3). The images are organized in three folders: train, validation and test. Deep Learning models, or to be more specific, Convolutional Neural Networks (CNNs) have proven to be really effective in a wide variety of computer vision tasks.

## Aim

The aim of this project is performance a Convolutional Neural Network and a Convolutional Autoencoder with the malaria images dataset. The advantages of this kind of NN is the capacibility to detect the important features without any human supervision and this make possible classify propertly the images.
