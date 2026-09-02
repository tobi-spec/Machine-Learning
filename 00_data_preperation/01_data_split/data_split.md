# Data Split

## Train/test

Use part of the data for traning and part of the data for testing. Aim is, that the model never saw the test data
to get real, unbiased results

Dataset
┌──────────────────────────────────────────┬──────────┐
│              Training 80%                │ Test 20% │
└──────────────────────────────────────────┴──────────┘

## Train/Validation/Test
Split the data for training further into two subpopulations. The train data for actual training and the validation data to
calculate the hyperparameters on unseen data during the training. 

Dataset
┌─────────────────────────────┬──────────────┬──────────┐
│        Training 70%         │ Validation   │ Test 15% │
│                             │     15%      │          │
└─────────────────────────────┴──────────────┴──────────┘

# K-fold cross-validation
Split train data into k folds, k-1 are used for training the last one is for testing. By using all combinations of folds
you get a range in where your model performance. 

Fold 1: TEST  Train Train Train Train
Fold 2: Train TEST  Train Train Train
Fold 3: Train Train TEST  Train Train
Fold 4: Train Train Train TEST  Train
Fold 5: Train Train Train Train TEST

could gives these metrics
Fold 1 → 0.84
Fold 2 → 0.81
Fold 3 → 0.86
Fold 4 → 0.83
Fold 5 → 0.85


So mean performance of the model is 0.838


Overall Workflow: 
                    Original data
                         │
                   Train / Test
                    80%     20%
                     │       │
                     │       └── untouched final test
                     │
                5-fold CV
                     │
             choose model +
             hyperparameters
                     │
               train on full
               training data
                     │
                     ▼
              evaluate once
              on test set


## Stratified Split - for classification

In classifiaction task it can happen that one class as significaten more data then the other

95% → class 0
 5% → class 1

A completely random split might accidentally produce:

Training set:
    96% class A
     4% class B

Test set:
    91% class A
     9% class B

scikit-learn has the parameter "stratify" for this problem