# Pytorch implementation of Deep Innovation Protection (DIP)

Paper: Risi and Stanley, "Deep Innovation Protection: Confronting the Credit Assignment Problem in Training Heterogeneous Neural Architectures ""
Proceedings of the Thirty-Fith AAAI Conference on Artificial Intelligence (AAAI-2021)

https://arxiv.org/abs/2001.01683


## Prerequisites

The code is partly based on the PyTorch implementation of "World Models" (https://github.com/ctallec/world-models).

Code requieres Python3 and PyTorch (https://pytorch.org). The rest of the requirements are included in the [requirements file](requirements.txt), to install them:
```bash
pip3 install -r requirements.txt
```

## Running the program

The world model is composed of three different components: 

  1. A Variational Auto-Encoder (VAE)
  2. A Mixture-Density Recurrent Network (MDN-RNN)
  3. A linear Controller (C), which takes both the latent encoding and the hidden state of the MDN-RNN as input and outputs the agents action

In contrast to the original world model, all three components are trained end-to-end through evolution. To run training:

```bash
python3 main.py
```

To test a specific genome:

```bash
python3 main.py --test best_1_1_G2.p
```

Additional arguments for the training script are:
* **--folder** : The directory to store the training results. 
* **--pop-size** : The population size.
* **--threads** : The number of threads used for training or testing.
* **--generations** : The number of generations used for training.
* **--inno** : 0 = Innoviation protection disabled. 1 = Innovation protection enabled. 


### Notes
When running on a headless server, you will need to use `xvfb-run` to launch the controller training script. For instance,
```bash
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 main.py
```

When running with a discrete VAE, the size of the latent vector is increased to 128 from the 32-dimensional version used for the standard VAE.

## Authors

* **Sebastian Risi**

