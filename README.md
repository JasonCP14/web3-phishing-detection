# Web3 Phishing Detection

This file includes the requirements, as well as the steps that you need to do in order to run the phishing detection Flask app.

## 1. Pre-requisites 

To execute the training pipeline, you would need a training dataset, with the following specifications:
- It is a csv file with 2 columns: `Messages` and `gen_label`. `Messages` contains all phishing and non-phishing that will be fitted into the model, while the `gen_label` contains 0 and 1 values for non-phishing and phishing respectively. 
- It must be located in the `/docker/backend/data/` directory alongside the given `test.csv`.
- It must be named as `train.csv`.

Besides that, there is a `config.json` file in the `/docker/backend` directory which contains all the hyperparameters that can be modified in this training. 


## 2. Deploying and Training

Having prepared the training dataset, you should run `docker compose build` on the root directory where the `docker-compose.yaml` is located in order to build the containers, and compose them. This will trigger the training pipeline automatically, and it might take a few minutes for the containers to fully be ready for use.


## 3. Ready for Use

Finally, you can run the Flask application by running `docker compose up` on the same root directory. This will turn on the Flask application, which can be opened through this [link](http://127.0.0.1:5000/).

In this page, you can enter your own sentence for phishing detection, and the model will return the classification along with its probability.