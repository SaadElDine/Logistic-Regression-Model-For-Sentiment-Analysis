# Logistic Regression Model for Sentiment Analysis

![image](https://github.com/SaadElDine/SST-Text-Classification-Using-Logistic-Regression/assets/113860522/0cc6e3be-56e0-4420-92b6-8c1bdac4b4ad)


## Overview
This model uses logistic regression to perform sentiment analysis on movie reviews. It is trained on the Stanford Sentiment Treebank (SST) dataset, which contains movie reviews labeled with sentiment scores ranging from 0 to 1.

## Dataset
The SST dataset consists of movie reviews labeled with sentiment scores, where scores closer to 0 indicate negative sentiment and scores closer to 1 indicate positive sentiment. To simplify the classification task, the sentiment scores are mapped to five categories as follows:
- **0 to 0.2 (inclusive):** Very Negative
- **0.2 to 0.4 (inclusive):** Negative
- **0.4 to 0.6 (inclusive):** Neutral
- **0.6 to 0.8 (inclusive):** Positive
- **0.8 to 1.0 (inclusive):** Very Positive

## Model Architecture
- **Features:** Word bi-grams are used as features to represent each sentence in the movie reviews.
- **Model:** Logistic regression is employed as the classification model.
- **Implementation:** Implemented using NumPy for numerical computations.
- **Components:**
  - **Classification Function (`predict`):** Uses the sigmoid function to compute the estimated class probabilities.
  - **Loss Function (`compute_loss`):** Utilizes binary cross-entropy loss to measure the difference between predicted and actual labels.
  - **Optimization (`sgd`):** Implements stochastic gradient descent (SGD) to update the weights of the model based on the computed gradients.

## Training Process
- **Training Dataset:** The model is trained on the training dataset from the SST dataset.
- **Training Algorithm:** Stochastic gradient descent (SGD) is used with a learning rate of 0.01 and 100 epochs.
- **Feature Extraction:** Word bi-grams are extracted from the text data to create the feature matrix.

## Testing and Evaluation
- **Testing Dataset:** The model is evaluated on the test dataset from the SST dataset.
- **Performance Metric:** Accuracy is used as the performance metric to evaluate the model's classification accuracy.
- **Accuracy:** The final accuracy achieved on the test set is 0.75.

## Conclusion
The logistic regression model demonstrates reasonable accuracy in classifying the sentiment of movie reviews on the SST dataset. Further improvements could be explored by incorporating more advanced features or using more complex models such as neural networks.
