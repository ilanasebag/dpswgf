# Differentially Private Gradient Flow based on the Sliced Wasserstein Distance


In this project, we introduce a novel differentially private generative modeling approach based on a
gradient flow in the space of probability measures. To this end, we define the gradient flow
of the Gaussian-smoothed Sliced Wasserstein Distance, including the associated stochastic
differential equation (SDE). By discretizing and defining a numerical scheme for solving
this SDE, we demonstrate the link between smoothing and differential privacy based on a
Gaussian mechanism, due to a specific form of the SDE’s drift term. We then analyze the
differential privacy guarantee of our gradient flow, which accounts for both the smoothing and
the Wiener process introduced by the SDE itself. Experiments show that our proposed model
can generate higher-fidelity data at a low privacy budget compared to a generator-based
model, offering a promising alternative.


### Overview of the repository

This repository contains two folders: 

• DPSWflow (which contains both versions of our mechanism: DPSWflow and DPSWflow-r): DP SW gradient flow.

• DPSWgen: generator based model trained with a DP loss (the DP SW).


### Running the project

To maintain a suitably low input space dimension, in order to mitigate the curse of dimensionality and reduce computational cost, DPSWflow(-r) and DPSWgen are preceded by an autoencoder. Subsequently, they take the latent space of the autoencoder as the input space, and ensure differential privacy using the DP gradient flow / generator. Therefore, we first have to train this auto-encoder


##### MNIST

• Train the AE and save the weights: python3.9 -m autoencoder.train --device 0 --config ~/config/mnist.yaml --data_path ~/dataset/MNIST/raw --exp_path ~/results_mnist_ae

• DPSWflow(-r): python3.9 -m flow.main --device 0 --config ~/config/mnist.yaml --data_path ~/dataset/MNIST/raw --exp_path ~/results_mnist_dpswflow

• DPSWgen: python3.9 -m model.main --config ~/config/mnist.yaml --data_path ~/dataset/MNIST/raw --exp_path ~/results_mnist_dpswgen

##### FashionMNIST

• Train the AE and save the weights: python3.9 -m autoencoder.train --device 0 --config ~/config/fmnist.yaml --data_path ~/dataset/FashionMNIST --exp_path ~/results_fmnist_ae

• DPSWflow(-r): python3.9 -m flow.main --device 0 --config ~/config/fmnist.yaml --data_path ~/dataset/FashionMNIST --exp_path ~/results_fmnist_dpswflow

• DPSWgen: python3.9 -m model.main --config ~/config/fmnist.yaml --data_path ~/dataset/FashionMNIST --exp_path ~/results_fmnist_dpswgen


#### CelebA

• Train the AE and save the weights: python3.9 -m autoencoder.train --device 0 --config ~/config/celeba.yaml --data_path ~/dataset/celeba --exp_path ~/results_celeba_ae

• DPSWflow(-r): python3.9 -m flow.main --device 0 --config ~/config/celeba.yaml --data_path ~/dataset/celeba --exp_path ~/results_celeba_dpswflow

• DPSWgen: python3.9 -m model.main --config ~/config/celeba.yaml --data_path ~/dataset/celeba --exp_path ~/results_celeba_dpswgen



