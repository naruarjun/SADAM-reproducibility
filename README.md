<h3 align="center">SAdam Reproducibility Study</h3>

 Reproducibility study carried out on the paper <a href = "https://openreview.net/forum?id=rye5YaEtPr">SAdam: A Variant of Adam for Strongly Convex Functions </a>, 
as part of the ML Reproducibility Challenge 2020 
    <br />
    <p align = "center"><a href="https://github.com/naruarjun/SADAM-reproducibility/issues">Request Feature/Report Bug</a>
    </p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#installation">Installation</a></li>
    <li>
        <a href="#usage">Usage</a>
        <ul>
            <li><a href="#neural-network-experiments">Neural Network Experiments</a></li>
            <li><a href="#regret-experiments">Regret Experiments</a></li>
            <li><a href="#use-the-optimizers">Use the Optimizers</a></li>
        </ul>
    </li>
    <li><a href="#reports">Results</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/naruarjun/SADAM-reproducibility.git
   ```
2. Install the requirements
   ```sh
   pip install -r requirements.txt 
   ```
<!-- USAGE EXAMPLES -->
## Usage

Here give a listing of all the command line arguments, for train.py 
and then write a line about the fact that samples are shown below. 

Arguments:
```
 --dataset 	    The dataset to be used in the experiment  (mnist(default)/cifar10/cifar100) 

 --lr               Learning rate for the optimizer           (default - 1e-3)
 
 -batch_size        Batch size for dataloader                 (default - 64)
 
 --model            Model architecture                        (logistic(default)/nn/resnet18)
 
 --epochs           No. of epochs to train the model          (default - 100)
 
 --optimizer        Optimizer to be used                      (adam(default)/amsgrad/scrms/scadagrad/ogd/sadam)
 
 --convex           To use convex version of optimizer or not (True/False(default))
 
 --decay            Regularization Factor                     (Default - 1e-2)
 
```

### Neural Network Experiments
A Sample way to execute is given below, however the parameters can be varied as per the user's wish, to generate all kinds of permutations with models, hyperparameters, datasets, optimizers and batch sizes. 
```sh
   python3 train.py --dataset mnist --lr 0.001 --batch_size 64 --decay 0 --optimizer adam --epochs 100 --model nn
```
### Regret Experiments
All the options mentioned above, can be used to run the regret experiments as well, however the *model* chosen should be *logistic* and *convex* parameter should be *True*. A sample execution is shown below - 
```sh
    python3 train.py --dataset mnist --lr 0.001 --batch_size 64 --decay 1e2 --optimizer adam --epochs 100 --model logistic --convex True
```
### Use the Optimizers
Code to import the optimizers. Once the optimizers are imported, one can use these optimizers, just like the standard ones provided by PyTorch are used with ```optimizer.zero_grad()``` and ```optimizer.step()``` whenever necessary. 
```sh
    import custom_optimizers as OP 
    """
    params - model.parameters() 
    lr - learning rate to be used 
    weight_decay - non zero value, in case some 
    convex - True / False, depending on the model to train
    """
    optimzer = OP.SC_RMSprop(params, lr=lr, weight_decay=decay, convex=convex)
```

## Results 

Below are the links to our corresponsing projects on Weights and Biases

<a href="https://wandb.ai/naruarjun/sadam-mnist-final">MNIST regret analysis</a>

<a href="https://wandb.ai/naruarjun/sadam-cifar10-final">CIFAR-10 regret analysis</a>

<a href="https://wandb.ai/naruarjun/sadam-cifar100-final">CIFAR-100 regret analysis</a>

<a href="https://wandb.ai/yashsarrof/mnist">MNIST 4 layer CNN analysis</a>

<a href="https://wandb.ai/yashsarrof/cifar10">CIFAR10 4 layer CNN analysis</a>

<a href="https://wandb.ai/yashsarrof/cifar100">CIFAR100 4 layer CNN analysis</a>

<a href="https://wandb.ai/yashsarrof/ResNet">CIFAR10 ResNet-18 analysis</a>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/Feature`)
3. Commit your Changes (`git commit -m 'Add some Feature'`)
4. Push to the Branch (`git push origin feature/Feature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Narayanan Elavathur Ranganatha - [@naruarjun1](https://twitter.com/naruarjun1) - naruarjun@gmail.com

Yash Raj Sarrof [@yashYRS](https://twitter.com/yashYRS) - yashrajsarrof18121998@gmail.com



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [The Authors of SAdam - Guanghui Wang, Shiyin Lu, Quan Cheng, Wei-wei Tu, Lijun Zhang](https://openreview.net/forum?id=rye5YaEtPr)
* [The ML Reproducibility Challenge 2020 Organising committee](https://paperswithcode.com/rc2020)
* []()
