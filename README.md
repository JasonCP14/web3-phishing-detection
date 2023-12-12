# Web3 Phishing Detection

This file includes the requirements, as well as the steps that you need to do in order to run the phishing detection Flask app.

## 1. Pre-requisites 

To execute the training pipeline, you would need a training dataset, with the following specifications:
- It is a csv file with 2 columns: `Messages` and `gen_label`. `Messages` contains all phishing and non-phishing that will be fitted into the model, while the `gen_label` contains 0 and 1 values for non-phishing and phishing respectively. 
- It must be located in the `/data/` directory alongside the given `test.csv`.
- It must be named as `train_data.csv`.

Besides that, there is a `config.json` file in the `/src/` directory which contains all the hyperparameters that can be modified in this training. 

## 2. Training

Having prepared the training dataset, you can start the training process by running the `py src/train_script.py` command, and this might take a few minutes to finish. This will create a model file inside the `docker/backend/saved_model` directory. 

## 3. Testing

To locally evaluate the results of the trained model, you can run `py /src/test_script.py`. This will load the model and run it on the test data, which is a file named `test_data.csv` located inside the `/data/` directory. The current result metrics is as follows:

|                     | Predicted Positive | Predicted Positive |
|---------------------|--------------------|--------------------|
| **Actual Negative** | 23                 | 10                 |
| **Actual Positive** | 1                  | 67                 |

## 4. Building

Now, after collecting all the required components for the container, you should run `docker compose build` on the root directory, where the `docker-compose.yaml` is located in order to build the containers, and compose them.

## 5. Deploying

Then, you can start all the containers application by running `docker compose up` on the same root directory. This will automatically deploy the Flask application, which can be opened through this [link](http://127.0.0.1:5000/).

## 6. Ready to Use

In this Flask page above, you can enter your own sentence for phishing detection, and the model will return the classification along with its probability.