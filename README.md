# Breast-Cancer
Feature selection by SVM,RF and NN on Breast Cancer data
Implementation Technical Notes
For the given assignment, Jupyter Notebook and Python 3.8 are used as the main development environment. In addition, sickit-learn (version 1.1.2) is used as our machine learning toolkit and matplotlib (version 3.6.0) as a visualization tool. It is also notable that pandas (version 1.5.0) was our means to provide necessary data structures. For each dataset, we split the data randomly into train and test with the ratio of 0.7 and 0.3 respectively. No portion of data is was considered as validation data because it was not asked to do so in the assignment notes. Regarding the feature selection libraries, the information gain method from sickit- learn and mRMR from their official pypi (version 0.2.5) package was used.
Performance Summary for Three Classifiers (Accuracy in percentage)
Table 1 Performance Summary on Different Datasets
a) Abstract
Basically, for predicting a class label for a record in the Decision Tree algorithm, we start from the root of the tree and compare the values of the root attribute with the record’s attribute. Based on the comparison, we follow the branch corresponding to that value and jump to the next node. Besides the decision tree, we have another classifier named Random Forest, which consists of many decision trees. It uses bagging and feature randomness when building each tree to try to create an uncorrelated forest of trees whose prediction is more accurate than that of any tree. Finally, in our Neural Network architecture, we use Multi-layer Perceptron which trains using backpropagation. More precisely, it trains using some form of gradient descent, and the gradients are calculated using backpropagation. For classification, it minimizes the Cross- Entropy loss function, giving a vector of probability estimates P(y|x) per sample x.
b) Introduction
b-1) Decision Tree and Random Forest: Decision Trees and Random Forests are generally used for classifying non-linearly separable data because they are also non-linear models. However, they can create both linear and nonlinear boundaries and this is because of how they cluster the data based on nested "if- else" statements. These statements draw vertical/horizontal lines between the samples and cluster them in rectangles. Consecutively, rectangles of the same class can be far away from each other (with other class rectangles in between) but still, they belong to the same class. This is how nonlinear relations are modeled.
It is notable that, random forest is an ensemble learning method. Ensemble learning is a general meta approach to machine learning that seeks better predictive performance by combining the predictions from multiple models. The three main classes of ensemble learning methods are bagging, stacking, and boosting.
Random forest uses bagging. Bootstrap aggregation, or bagging for short, is an ensemble learning method that seeks a diverse group of ensemble members by varying the training data. This typically involves using a single machine learning algorithm, almost always an unpruned decision tree, and training each model on
Method Circles0.3
Moons1 Halfkernel Spiral1 Twogaussian33
Twogaussian42
Decision Tree
98.66
99.33
99
99
97.66
93
Random Forest
98.66
99.33
99
98.66
99.33
93
Neural Network
100
99.66
100
99.33
98.66
93.66
3
a different sample of the same training dataset. The predictions made by the ensemble members are then combined using simple statistics, such as voting or averaging.
From an accuracy point of view, generally, the random forest gives more accurate and precise results because it avoids and prevents overfitting by using multiple trees and it can generalize the data in a better way. This randomized feature selection makes a random forest algorithm much more accurate than a decision tree. However, as shown in the accuracy table, in these datasets we only have two features which are x and y. So we do not see much difference between random forest and decision tree in the accuracy of the classification and both of them work well in both linear and non-linear datasets. From the time complexity point of view, since a random forest combines several decision trees, it is a long process, yet slow.
Finally, explainability is one of the main positive points of tree-based methods.
b-2) Multi-layer Perceptron
The main reason behind choosing this type of neural networks by our group was flexibility and simplicity. Choosing the suitably simple tools for solving a problem should be considered since we are dealing with computational and memory limitations.
Neural Networks are getting better when the complexity of the dataset increases. In fact, they are effective in nonlinear spaces where the structure of the relationship is not linear. That is why they have high accuracy in datasets other than two Gaussians which are non-linear. (They do not have high accuracy on two Gaussians because these two datasets are linearly separable ones). More specifically, without activation functions, neural networks can only learn linear relationships. In order to fit curves, we'll need to use activation functions. If the final (output) layer is a linear unit (meaning, no activation function), that makes the network appropriate to a regression task, where we are trying to predict some arbitrary numeric value. Other tasks (like classification) might require an activation function on the output.
c) Experiments and Discussion
As seen above in Table1, all three of our classifiers obtained very good results on all of the datasets. If we compare these results to previous assignments, we can see that overall results were better and each classifier generalized better to different datasets.
From the comparison point of view, Random Forests not only achieve similarly good performance results in practical applications, but they also have some advantages compared to Neural Networks in specific cases including their robustness as well as benefits in cost and time. This is because of the long training time and the need for high-quality hardware for processing in the case of using Neural Networks.
However, all of these three classifiers (Decision Tree, Random Forest, Neural Network) work better for non-linearly separable datasets which are all the datasets except the Gaussians.
Finally, when we have excellent results with both neural network and classic machine learning methods it is preferable to use classic machine learning methods because they are interpretable unlike neural networks that considered as black-boxed methods. This phenomenon gains more and more attention everyday since it is necessary to find out about the basis of an automated decision-making process for each decision. Black- boxed methods are a place of controversy especially in field of medical diagnosis. This is of the main reasons that most of the recent papers published in Nature or etc. combine their neural based methods with attention models such as SHAP or Grad-CAM to improve interpretability of their approach.
4

c-2) Choosing Parameters for Multi-layer Perceptron
• Hidden layer sizes: This parameter allows us to set the number of layers and the number of nodes we wish to have in the Neural Network Classifier. Each element in the tuple represents the number of nodes at the ith position where i is the index of the tuple. Thus, the length of the tuple denotes the total number of hidden layers in the network and each integer gives you the size of the layers. In scikit-learn, it has a single hidden layer with 100 units as default. Based on the (10, 10) value in our architecture, here we have two hidden layers of size 10. We had to choose small number of neurons and hidden layers because of three main reasons. First, we did not have many features in these datasets. Second, rule of thumb suggests that the more hidden layers, the more data is necessary for the neural network to converge. Finally, we have to be careful about the overfitting phenomenon since we know it happens more easily in more complex models.
• Max iter: It denotes the number of epochs which is different for each dataset. We determine this parameter by trial error to find the point that our optimizer converge. The list of required iterations for each dataset in our experiment goes as follows: [ Circle0.3: 400, Halfkernel: 400, Moons:500, Spiral: 700, Twogaussian33: 250, Twogaussian42: 600]
• Activation: This is the activation function for the hidden layers. An activation function is basically a simple function that transforms its inputs into outputs that have a certain range. There are various types of activation functions that perform this task in different manners. In our architecture, we use the ReLU activation function which returns 0 if it receives any negative input, but for any positive value x, it returns that value back. Hence, it gives an output that has a range from 0 to infinity.
• Solver: This parameter specifies the algorithm for weight optimization across the nodes. Our solver is adam which refers to a stochastic gradient-based optimizer.
• Learning rate: The amount that the weights are updated during training is referred to as the step size or the “learning rate.” In our architecture, the value is adaptive which keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol (tol is Tolerance for the optimization, and the default value is 1e-4), or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.
d) Feature Selection and Multi-class Classification on Breast Cancer Dataset
Feature selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model. It is notable that feature selection is different from dimension reduction techniques.
d-1) mRMR: mRMR, which stands for "minimum Redundancy - Maximum Relevance", is a feature selection algorithm. The peculiarity of mRMR is that it is a minimal-optimal feature selection algorithm. This means it is designed to find the smallest relevant subset of features for a given Machine Learning task. Selecting the minimum number of useful features is desirable for many reasons: memory consumption, time required, performance, explainability of results.
d-2) Information Gain: Information gain calculates the reduction in entropy or surprise from transforming a dataset in some way. It is commonly used in the construction of decision trees from a training dataset, by evaluating the information gain for each variable, and selecting the variable that maximizes the information gain, which in turn minimizes the entropy and best splits the dataset into groups for effective classification.
5

Information gain can also be used for feature selection, by evaluating the gain of each variable in the context of the target variable. In this slightly different usage, the calculation is referred to as mutual information between the two random variables.
For the breast cancer dataset, we got around 1380 features and 158 samples. Obviously, we have to consider feature selection to be able to create a model to classify our samples correctly. Since we have a computational limitation (especially for this assignment which is done on personal computers) we have to make a trade-off between maximizing the performance and minimizing the features being used to represent dataset’s variance. Hence, by trial and error we found that by using best 20 features in each of these feature selection methods we are able to get proper results. Although, ‘proper’ is somehow subjective and in terms of classification of medical related problems sometimes 95 percent is considered average because wrong decisions usually have a greater cost rather than a problem such as spam filtering. Also, by multiplying 5 percent in the number of patients, we will see that the number of errors will not be that small. So, here by ‘proper’ we mean in terms of our hardware limitations and a class assignment.
Then we combined these feature selection methods with SVM and grid search to find the best classifier for the mentioned dataset and set of selected features.
This is our search space for grid search over different SVM classifiers:
• kernel: ('linear', 'poly', 'rbf')
• degree: range(2, 6)
• gamma: ('scale', 'auto')
• C: [0.001, 0.01, 0.1, 1]
d-3) One vs All and All vs All
• One vs All: In One-vs-All classification, for the N-class instances dataset, we have to generate the N-binary classifier models. The number of class labels present in the dataset and the number of generated binary classifiers must be the same. In this method, for each class we consider that we are solving a binary classification. Our first label will be the class that we selected in that step, and all the other samples in different classes will be considered as our second class. This is why this method is called One vs All or One vs Rest (ovr)
• All vs All: In All vs All or One-vs-One classification, for the N-class instances dataset, we have to generate the N* (N-1)/2 binary classifier models. Using this classification approach, we split the primary dataset into one dataset for each class opposite to every other class.
Choosing between OVA or OVR
So, we can conclude that One vs All multi-class classification is challenging when we deal with large datasets having many numbers of class instances. Because we generate that many classifiers model and to train to those models, we create that many input training datasets from the primary dataset. On the other hand, In All vs. All multi-class classification, we split the primary dataset into one binary classification dataset for each pair of classes. So, we need acceptable number of samples in each class to get a proper
6

result while using this method. Since our dataset contains 5 classes and 158 samples (before splitting into train and test) it will be a better choice to use One vs All method for multi-class classification.
d-4) SVM and mRMR
Best classifier parameters: 'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'
d-5) SVM and Information Gain
Best classifier parameters: 'C': 0.1, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'
Overall, we can see that results obtained from mRMR features slightly edge the IG method. While these performance results would be considered acceptable in many fields. In the area of medicine and medical diagnosis, they are considered as average. Although we could get way better results on this dataset, our main issue was hardware limitations as explained above.
  Metrics
   Best SVM using mRMR Best SVM using IG
  specificity
   95.18
   94.75
  sensitivity
 80.99
 79.29
  accuracy
   92.50
   91.66
  ppv
 82.94
 77.52
  npv
   95.61
   94.76
 7

References:
[1] Hastie, T., Tibshirani, R., Friedman, J. (2001). The Elements of Statistical Learning. New York, NY, USA: Springer New York Inc.
[2] Aurlien Gron. 2017. Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (1st. ed.). O'Reilly Media, Inc.
[3] Wasserman, L. (2010). All of statistics : a concise course in statistical inference. New York: Springer. ISBN: 9781441923226 1441923225
[4] Ding, Chris, and Hanchuan Peng. "Minimum redundancy feature selection from microarray gene expression data." Journal of bioinformatics and computational biology 3.02 (2005): 185-205.
