# Machine Learning by Andrew Ng
 ## Machine Learning course offered by Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning)

 ### About this Course

Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI.

This course provides a broad introduction to machine learning, datamining, and statistical pattern recognition. Topics include: (i) Supervised learning (parametric/non-parametric algorithms, support vector machines, kernels, neural networks). (ii) Unsupervised learning (clustering, dimensionality reduction, recommender systems, deep learning). (iii) Best practices in machine learning (bias/variance theory; innovation process in machine learning and AI). The course will also draw from numerous case studies and applications, so that you'll also learn how to apply learning algorithms to building smart robots (perception, control), text understanding (web search, anti-spam), computer vision, medical informatics, audio, database mining, and other areas.

<br />

### Skills You Will Gain

`Logistic Regression`&nbsp;   |&nbsp;   `Artificial Neural Network`&nbsp;   |&nbsp;   `Machine Learning (ML) Algorithms`&nbsp;   |&nbsp;   `Machine Learning`
<br />

### Syllabus

> #### WEEK 1
> __Introduction__
> Welcome to Machine Learning! In this module, we introduce the core idea of teaching a computer to learn concepts using data—without being explicitly programmed. The Course Wiki is under construction. Please visit the resources tab for the most complete and up-to-date information.
>
> **15. Graded:** Introduction
>
> __Linear Regression with One Variable__
> Linear regression predicts a real-valued output based on an input value. We discuss the application of linear regression to housing price prediction, present the notion of a cost function, and introduce the gradient descent method for learning.
>
> **16. Graded:** Linear Regression with One Variable
>
> __Linear Algebra Review__
> This optional module provides a refresher on linear algebra concepts. Basic understanding of linear algebra is necessary for the rest of the course, especially as we begin to cover models with multiple variables.
>
> **14. Practice Quiz:** Linear Algebra
>

<br />

> #### WEEK 2
> __Linear Regression with Multiple Variables__
> What if your input has more than one value? In this module, we show how linear regression can be extended to accommodate multiple input features. We also discuss best practices for implementing linear regression.
>
> **25. Graded:** Linear Regression with Multiple Variables
>
> __Octave/Matlab Tutorial__
> This course includes programming assignments designed to help you understand how to implement the learning algorithms in practice. To complete the programming assignments, you will need to use Octave or MATLAB. This module introduces Octave/Matlab and shows you how to submit an assignment.
>
> **8. Programming Assignment:** Linear Regression
> **9. Graded:** Octave/Matlab Tutorial
>
<br />

> #### WEEK 3
> __Logistic Regression__
>Logistic regression is a method for classifying data into discrete outcomes. For example, we might use logistic regression to classify an email as spam or not spam. In this module, we introduce the notion of classification, the cost function for logistic regression, and the application of logistic regression to multi-class classification.
>
> *16. Graded:** Logistic Regression
>
> __Regularization__
> Machine learning models need to generalize well to new examples that the model has not seen in practice. In this module, we introduce regularization, which helps prevent models from overfitting the training data.
>
> **10. Programming Assignment:** Logistic Regression
> **11. Graded:** Regularization
>
<br />

> #### WEEK 4
> ___Neural Networks: Representation__
> Neural networks is a model inspired by how the brain works. It is widely used today in many applications: when your phone interprets and understand your voice commands, it is likely that a neural network is helping to understand your speech; when you cash a check, the machines that automatically read the digits also use neural networks.
>
> **14. Programming Assignment:** Multi-class Classification and Neural Networks
> **15. Graded:** Neural Networks: Representation
>
<br />

> #### WEEK 5
> __Neural Networks: Learning__
> In this module, we introduce the backpropagation algorithm that is used to help learn parameters for a neural network. At the end of this module, you will be implementing your own neural network for digit recognition.
>
> **17.  Programming Assignment:** Neural Network Learning
> **18. Graded:** Neural Networks: Learning
>
<br />

> #### WEEK 6
> __Advice for Applying Machine Learning__-
> Applying machine learning in practice is not always straightforward. In this module, we share best practices for applying machine learning in practice, and discuss the best ways to evaluate performance of the learned models.
>
> **15. Programming Assignment:** Regularized Linear Regression and Bias/Variance
> **16. Graded: Advice for Applying Machine Learning
>
> __Machine Learning System Design__
> To optimize a machine learning algorithm, you’ll need to first understand where the biggest improvements can be made. In this module, we discuss how to understand the performance of a machine learning system with multiple parts, and also how to deal with skewed data.
>
> **9. Graded:** Machine Learning System Design
>
<br />

> #### WEEK 7
> __Support Vector Machines__
> Support vector machines, or SVMs, is a machine learning algorithm for classification. We introduce the idea and intuitions behind SVMs and discuss how to use it in practice.
>
> **8. Programming Assignment:** Support Vector Machines
> **9. Graded:** Support Vector Machines
>
<br />

> #### WEEK 8
> __Unsupervised Learning__-
> We use unsupervised learning to build models that help us understand our data better. We discuss the k-Means algorithm for clustering that enable us to learn groupings of unlabeled data points.
>
> **7. Graded:** Unsupervised Learning
>
> __Dimensionality Reduction___
> In this module, we introduce Principal Components Analysis, and show how it can be used for data compression to speed up learning algorithms as well as for visualizations of complex datasets.
>
> **9. Programming Assignment:** K-Means Clustering and PCA
> **10. Graded:** Principal Component Analysis
>
<br />

> #### WEEK 9
> __Anomaly Detection__
> Given a large number of data points, we may sometimes want to figure out which ones vary significantly from the average. For example, in manufacturing, we may want to detect defects or anomalies. We show how a dataset can be modeled using a Gaussian distribution, and how the model can be used for anomaly detection.
>
> **10. Graded:** Anomaly Detection
>
> __Recommender Systems__
> When you buy a product online, most websites automatically recommend other products that you may like. Recommender systems look at patterns of activities between different users and different products to produce these recommendations. In this module, we introduce recommender algorithms such as the collaborative filtering algorithm and low-rank matrix factorization.
>
> **8. Programming Assignment:** Anomaly Detection and Recommender Systems
> **9. Graded:** Recommender Systems
>
<br />

> #### WEEK 10
> __Large Scale Machine Learning__
> Machine learning works best when there is an abundance of data to leverage for training. In this module, we discuss how to apply the machine learning algorithms with large datasets.
>
> **8. Graded:** Large Scale Machine Learning
>
<br />

> #### WEEK 11
> __Application Example: Photo OCR__
> Identifying and recognizing objects, words, and digits in an image is a challenging task. We discuss how a pipeline can be built to tackle this problem and how to analyze and improve the performance of such a system.
>
> **7. Graded:** Application: Photo OCR
>
<br />
