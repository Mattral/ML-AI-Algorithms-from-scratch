# Supervised Learning Algorithms from Scratch

This repository contains Python implementations of various supervised learning algorithms built from the ground up. Each algorithm is implemented with detailed documentation to help you understand the underlying principles.

# Repo Structure

```
├── LICENSE
├── README.md          <- The top-level README for developers/collaborators using this project.
├── neural_network      <- Folder for Neural Network implementations
│   ├── AutoEncoder.py
│   ├── BoltzmannMachine.py
│   ├── GenerativeAdversarialNetwork.py
│   ├── HopfieldNetwork.py
│   ├── LongShortTermMemoryLSTM.py
│   ├── MultiLayerPerceptronClassification.py
│   ├── MultiLayerPerceptronRegression.py
│   ├── RadialBasisFunctionNetworks.py
│   ├── SelfAttentionMechanism.py
│   ├── SimpleCNN.py
│   ├── SimpleEncoderDecoder.py
│   ├── SimpleRNN.py
│   ├── SingleLayerPerceptronClassification.py
│   ├── SingleLayerPerceptronRegression.py
│   ├── TitanicSurvialBySingleLayerPerceptron.py
│   └── Transformer.py
│
├── reinforcement_learning  <- Folder for Reinforcement Learning implementations
│   ├── Deep Deterministic Policy Gradients
│   ├── Deep Q Network
│   ├── Proximal Policy Optimization
│   └── QLearning
│
├── supervised            <- Folder for Supervised Learning implementations
│   ├── DecisionTrees
│   ├── KnearestNeighbour
│   ├── LassoRegression
│   ├── LinearRegression
│   ├── LogisticRegression
│   ├── Naive Bayes
│   ├── RandomForest
│   ├── RidgeRegression
│   └── SupportVectorMachines
│
├── unsupervised          <- Folder for Unsupervised Learning implementations
│   ├── AprioriAlgorithm
│   ├── Density-Based Spatial Clustering of Applications with Noise
│   ├── Expectation-Maximization
│   ├── Gaussian Mixture Model
│   ├── HierarchicalClustering
│   ├── IndependentComponentAnalysis
│   ├── K-MedoidsClustering
│   ├── KMeansPlusPlus
│   ├── PrincipalComponentAnalysis
│   ├── SelfOrganizing Map
│   ├── kmeanclustering
│   └── tSNE
│
└── 

```

## Algorithms Included:

1. **Decision Trees**
   - File: `DecisionTreeClassification.py+DecisionTreeRegression.py+unitTestClassifier.py+unitTestRegressor.py`
   - Description: Implementation of decision trees, a popular algorithm for classification and regression tasks. Decision trees make decisions based on a series of questions.

2. **K-Nearest Neighbors (KNN)**
   - File: `KNNClassifier.py+KNNRegressor.py`
   - Description: Implementation of KNN, a simple and versatile algorithm for classification and regression. It classifies data points based on the majority class among their k-nearest neighbors.

3. **Lasso Regression**
   - File: `LassoRegression.py`
   - Description: Implementation of Lasso regression, a linear regression technique that includes an L1 regularization term. Lasso can be used for feature selection.

4. **Linear Regression**
   - File: `LinearRegressionAlgo.py+unittestLR.py`
   - Description: Implementation of linear regression, a fundamental algorithm for predicting a continuous outcome based on linear relationships between input features.

5. **Logistic Regression**
   - File: `LogisticRegressionAlgo.py+LogisticRegressionTitanicDataset.py`
   - Description: Implementation of logistic regression, a classification algorithm that models the probability of a binary outcome. It's widely used for binary classification tasks.

6. **Naive Bayes**
   - File: `NaiveBayesClassifier.py`
   - Description: Implementation of Naive Bayes, a probabilistic classifier based on Bayes' theorem. It assumes that features are conditionally independent.

7. **Random Forest**
   - File: randomForestClassification.py+randomForestRegression.py`
   - Description: Implementation of random forests, an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes.

8. **Ridge Regression**
   - File: `RidgeRegression.py`
   - Description: Implementation of Ridge regression, a linear regression technique that includes an L2 regularization term. Ridge regression can help prevent overfitting.

9. **Support Vector Machines (SVM)**
   - File: `LinearSVMClassifier.py+SVMRegressor.py`
   - Description: Implementation of SVM, a powerful classification algorithm that finds the hyperplane that best separates different classes. It can handle both linear and non-linear relationships.

## Usage

Each algorithm is provided as a standalone Python script. You can run these scripts to see the algorithms in action. Additionally, the code is extensively documented to help you understand the implementation details.


## Underlying Technology

The algorithms are implemented using Python and NumPy. The code is designed for educational purposes to help users learn about supervised learning concepts and algorithms.

