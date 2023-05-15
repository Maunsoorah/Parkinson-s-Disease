# Parkinson's Disease
Parkinson's Disease Classification Using Machine Learning: A Comprehensive Analysis of Biomedical Voice Measurements

#### Introduction

Parkinson's disease (PD) is a neurodegenerative disorder that affects millions of people worldwide, causing motor and non-motor symptoms that significantly impact their quality of life. Timely and accurate diagnosis of PD plays a crucial role in effective disease management and personalized treatment. In recent years, machine learning algorithms have emerged as valuable tools for analyzing biomedical data and aiding in disease classification.

The dataset used is from the Parkinson's dataset from the UCI data repository (archive.ics.uci.edu) and encompasses an extensive collection of voice recordings from 31 individuals, including 23 with PD. Each row in the dataset represents a unique voice recording, while each column corresponds to specific voice measurements.

The primary objective of this study is to employ the K-nearest neighbors (KNN) classification algorithm to accurately classify individuals as either PD-positive or PD-negative based on their voice measurements. KNN is a popular supervised learning algorithm that assigns labels to data points based on the similarity of their features to those of previously labeled data points. By leveraging this algorithm, I have built a predictive model that can effectively distinguish between PD and non-PD cases.

#### Attribute Information/Column names description

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/7bdfd434-bf30-471f-9a66-d55c606b99b9)

#### Exploratory Data Analysis

Import the required libraries

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/3dbb2e1c-2dcf-4eb4-83e6-427870be188a)

Load the Dataset

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/edb37873-3d2e-4dbf-801b-8213f7b199fd)

View the dataset

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/145d1393-e744-4b18-96e2-0bf37eb9450d)

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/e7af8077-0e94-4c8d-a051-63c4d1abc2a6)

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/c10bde4c-5cf9-4cf0-a73c-8f72256bbb54)

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/708f7995-500e-457f-a4b9-eb65ecb6a2ce)

Using the .describe(), we will view the statistical details of the dataset

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/e8f93d90-83ff-484a-bbc6-386c58b9d246)

Let's check for missing valuses

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/dd5b65b8-597e-46e8-b8dd-bbb427fe8822)

This shows that we do not have any missing values and we can proceed with our model.

#### Classification using K-Nearest Neighbor (KNN)

According to medium.com, “k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.” While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.
First, we will check for the relationship between the attributes of the dataset by plotting a Correlation matrix. We are changing the value of the fmt function from the default .2f to .1f so that the matrix displays figures with one decimal place as this will enhance readability. 
•	-1 indicates a perfectly negative linear correlation between two variables
•	0 indicates no linear correlation between two variables
•	1 indicates a perfectly positive linear correlation between two variables

The further away the correlation coefficient is from zero, the stronger the relationship between the two variables.
Looking at the matrix below, we can see that there is good correlation between the variables of the dataset.

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/600a0c2e-d048-4678-9f1d-1cc892dac6d9)

It is usually a good idea to visualize the distribution of our dependent variable which is the status column in this case as this will let us know if the data is evenly distributed or not.

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/47a0b580-34d1-458b-a24c-04b5337a758a)

In KNN, we need to choose input and output features for us to use in our model. For this data set, the 17th column with column name “status” is our dependent variable while all other columns (except the name column) are our input features.

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/c72170f0-1504-43a4-80bb-422cd0bcf928)

Overfitting is statistics can be defined as a modelling error which happens when some functions are too closely aligned to a set of datapoints.
Overfitting the model generally takes the form of making an overly complex model to explain idiosyncrasies in the data under study. In reality, the data often studied has some degree of error or random noise within it. Thus, attempting to make the model conform too closely to slightly inaccurate data can infect the model with substantial errors and reduce its predictive power.
In order to avoid overfitting we will be splitting our dataset to train set and test set. Spitting our dataset allows us to compare the performance of machine learning algorithms used for our predictive modeling problem.

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/4381d12d-e142-4e76-9d6d-b5e069cf75bd)

We need to check for the existence of constant and duplicate features i.e features with similar variables. These features are similar to other variables and provide no information that allows our model to predict appropriately.
The code below is used to check for constant features and dropped if found and the output shows that our dataset does not have any and no feature was dropped.

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/629b6f28-e53c-4bac-b2af-b63be6124610)

Feature Scaling and Model Training

Feature scaling is a method used to standardize the range of independent variables or features of data. The goal is to transform the data so that each feature is in the same range (e.g. between 0 and 1). This ensures that no single feature dominates the others, thereby negatively affecting the outcome of our model.

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/a07fbeee-564f-438a-8d52-618177188638)

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/796e3b75-2c93-4dba-905d-ce1ce2660bd2)

Finally, we will evaluate the performance of our model using score, Confusion matrix and Classification report.
A classification report can be used to measure the quality of predictions from a classification algorithm. How many predictions are true and how many are false.
In the classification report, we have 3 main components, ‘Precision’, ‘Recall’ and ‘f1-score’.
PRECISION – means what percentage of your predictions were correct i.e Accuracy of positive predictions.
RECALL – What percent of the positive cases were correctly picked. i.e Fraction of positives that were correctly identified.

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/2c11cd94-c896-486a-9414-db076e85bf1e)

![image](https://github.com/Maunsoorah/Parkinson-s-Disease/assets/114883368/37c38fb0-1f3a-461e-9be0-7b9a07bb3dc5)

#### Conclusion

We have used the K-nearest neighbors (KNN) classification algorithm, to classify Parkinson's disease (PD) based on voice measurements. 
The KNN model achieved high accuracy, precision, recall, and F1-score, demonstrating its effectiveness in distinguishing between PD-positive and PD-negative cases. The findings suggest that machine learning algorithms, like KNN, can be valuable tools for PD classification using voice measurements. 
However, further research on larger and more diverse datasets might be needed to validate and improve the model's performance. 




