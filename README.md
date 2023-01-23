# Measurable Energy Technical Challenge

## Summary

This project allows you to train your own [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network), and use it to create your own fashion items!

The project includes the following elements to meet the requirements of the technical challenge:
  
- README
- DataLoader w/ transforms
- Training script
- ML Model implementation
- Config file
- MLFlow experiment tracking and model registry
- Confusion matrix example for the discriminator model


## Decisions
- I chose to train a GAN model partly to mix things up, as I thought doing a simple classifier would be the obvious choice, but I think it also demonstrates all of the skills that would be needed to train a simpler model

- Although a GAN doesn't really use the more traditional 'train-validate-test' flow that most models do, I decided to include a validation step and also a testing step on the discriminator model, purely to demonstrate my understanding of the process
  
- I decided to use a 'vanilla' Pytorch implementation just to demonstrate lower-level control of the model training flow, which has better visibility for some of the less traditional aspects of training a GAN
  
- I used docker for the MLFlow server but not for the training code just to demonstrate communication between services outside of docker
  
- I haven't included unittests at this stage just due to time, but the `tests` folder exists just to show potential project organisation
  
- NOTE: In the data loading part of the code, during the data splitting, I extract a subset of the FashionMNIST subset just to speed up training times, this is why there is a significant chunk of the data assigned to a dummy variable incase this looked odd!


# Getting Started

## Pre-requisites
### Pytorch
Use the top-level requirements.txt file to install dependencies for the model training code. The code will use a GPU if available, but will run on a CPU if not.

### MLFlow
The MLFlow server will run on any machine with [Docker-compose](https://docs.docker.com/compose/) installed. (See [here](https://docs.docker.com/v17.09/engine/installation/) for guidance on Docker installation)

### MinIO
MinIO is a S3 replica that can easily be spun up with Docker, and is being used as the storage backend for MLFlow in this project. 

**When docker has started all of the services you must login to the console and create a bucket called 'mlflow'. Credentials are stored in the provided `.env` file** 

### **It will not work without this!**

## Data

The model is trained using the [MNIST Fashion](https://github.com/zalandoresearch/fashion-mnist) dataset, which is automatically downloaded prior to training, and saved at `/data/`.

## MLFlow

To spin up the MLFlow server components, from the root of the project directory:

```bash
sudo docker-compose up -d
```
At this point, please set the following environment variables so your training code knows where to send the collected tracking metrics and artifacts and provides access to S3 clone (MinIO):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5009
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
```

The MLFlow server can be found at `http://localhost:5009/` and the associated MinIO console, used for the artifact storage backend, can be found at `http://localhost:9001/`. All credentials are saved to version control in the `.env` file

All artifacts, including finished models, checkpoints and example images are all able to explored through the MLFlow UI. 

***The artifacts are also stored locally for the purposes of this challenge, incase you wanted to explore any of the files outside of the MLFlow UI***

Each run also stores the parameters used for training, as well as keeping track of metrics and provided graphical views over time.



## Train Model
To train the model, use the `train_model` script:

```bash
python3 gan_trainer/train_model.py
```

There is also a command line option to pass in a model checkpoint to resume a training run:

```bash
python3 gan_trainer/train_model.py --checkpoint=$CKPT_PATH
```

# Results
***I've used the best practices for reproducibility as suggested by the Pytorch documentation. The seed is declared in the config file.***

I ran the training script using the currently committed configuration for a total of 40 Epochs, this took approx. 2 hours on a single CPU on my laptop. I tried to reduce this as much as possible while still producing some semi-interesting results!

### Real images 
![Real Examples](examples/real_samples.png?raw=true)
### Example generated images:
#### **0 Epochs**
![Generated Examples](examples/gen_examples_epoch_0.png?raw=true)

#### **10 Epochs**
![Generated Examples](examples/gen_examples_epoch_10.png?raw=true)

#### **20 Epochs**
![Generated Examples](examples/gen_examples_epoch_20.png?raw=true)

#### **30 Epochs**
![Generated Examples](examples/gen_examples_epoch_30.png?raw=true)

#### **38 Epochs**
![Generated Examples](examples/gen_examples_epoch_38.png?raw=true)

## Discriminator testing confusion matrix (%)
![Confusion Matrix](examples/test_cm.png?raw=true)

## Discriminator testing accuracy score(%)
![Accuracy Score](examples/acc_score.png?raw=true)

## Training Discriminator Loss
![Training Discriminator Loss](examples/train_disc_loss.png?raw=true)

## Training Generator Loss
![Training Generator Loss](examples/train_gen_loss.png?raw=true)

## Validation Discriminator Loss
![Training Generator Loss](examples/val_disc_loss.png?raw=true)
