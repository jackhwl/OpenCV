# RubikCube
[The key differences between Python 2.7.x and Python 3.x with examples](https://sebastianraschka.com/Articles/2014_python_2_3_key_diff.html)

https://github.com/kkoomen/qbr
http://kociemba.org/computervision.html

MasterOpenCV code:
https://github.com/PacktPublishing/Mastering-OpenCV-4-with-Python

http://www.youngscientist.com.au/wp-content/uploads/2015/02/Physics-10-12-Justin-Marcellienus-report.pdf

tensorflow-1.15.0
pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.15.0-py3-none-any.whl
pip3 install tensorflow==1.15, keras==2.3.1 python 3.7
pip3 install 'h5py==2.10.0' --force-reinstall

https://www.easy-tensorflow.com/tf-tutorials/basics/graph-and-session

pyenv install 3.8.13

# Scikit-Learn Tutorial | Machine Learning With Scikit-Learn | Sklearn | Python Tutorial | Simplilearn
https://www.youtube.com/watch?v=0Lt9w-BxKFQ

# Sample code
https://github.com/ageron/handson-ml2

## Chapter 10
* pip3 install tensorflow
* tensorflow and keras

# Machine Learning with JavaScript
* feature vs label
* classification vs regression
1. Identify data that is relevant to the problem
    * 'Features' are categories of data points that affect the value of a 'label'
2. Assemble a set of data related to the problem you're trying to solve
    * Datasets almost always cleanup or formatting
3. Decide on the type of output you are predicting
    * Regression used with continuous values, classification used with discrete values
4. Based on type of output, pick an algorithm that will determine a correlation between your 'features' and 'labels'
    * Many, many different algorithms exist, each with pros and cons
5. Use model generated by algorithm to make a prediction
    * Models relate the value of 'features' to the value of 'labels'

# K-Nearest Neighbor (knn)
# Section 3: Onwards to Tensorflow JS!
* Tensor Shape and Dimension
* 36. Let's Get Our Bearings
* 37. A Plan to Move Forward
* 38. Tensor Shape and Dimension
* * 1D ~ 6D tensor
## Shape: how many records in each dimension, 2D Shape [#rows, #columns], calling .length once on each dimension from outside in
* 39. Elementwise Operations
```
const data = tf.tensor([1,2,3])
const otherData = tf.tensor([4,5,6])

data.add(otherData)
data.sub(otherData)
data.mul(otherData)
data.div(otherData)

```
* 40. Broadcasting Operations
  two tensor shape different, broadcasting works when take shape of both tensors from right to left,
  the sahpes are equal or one is '1'
* 41. Logging Tensor Data data.print()
* 42. Tensor Accessors data.get(0,1)
    1d accessor, 2d accessor, ... get only, no set
* 43. Creating Slices of Data
    data.slice([0, 2], [-1, 1]), start index and size
* 44. Tensor Concatenation
    tansorA.concat(tensorB, 1) 0 vertical (default), 1 horizontal
* 45. Summing Values Along an Axis
    ```
    const jumpData = tf.tensor([[71,72,73], 
                                [73,72,70], 
                                [66,65,67], 
                                [75,62,77]]);
    const playerData = tf.tensor([[1,170],
                                [2,163],
                                [3,180],
                                [4,175]]);

    jumpData.sum(1,1).concat(playerData,1)
    ```
* 46. Massaging Dimensions with ExpandDims
jumpData.sum(1).expandDims(1).concat(playerData,1)
## Section 4: Applications of Tensorflow
* 47. KNN with Regression 
* 48. A Change in Data Structure
* 49. KNN with Tensorflow
    ```
    const features = tf.tensor([
        [-121, 47],
    [-121.2, 46.5],
    [-122, 46.4],
    [-120.9, 46.7]
    ])

    const labels = tf.tensor([
    [200],
    [250],
    [215],
    [240]
    ])

    const predictionPoint = tf.tensor([-121, 47])

    features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims(1)
    ```
    * 50. Maintaining Order Relationships
    ```
    features.sub(predictionPoint).pow(2).sum(1).pow(0.5).expandDims(1).concat(labels,1)
    ```
    * 51. Sorting Tensors
    ```
    .unstack()
    .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
    ```
    * 52. Averaging Top Values
    ```
    .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
    .slice(0, k)
    .reduce((acc, pair) => acc + pair.get(1), 0) / k
    ```
    * 53. Moving to the Editor
    * 54. Loading CSV Data
    * 55. Running an Analysis
    * 56. Reporting Error Percentages
    * 57. Normalization or Standardization?
    * 58. Numerical Standardization with Tensorflow
    ```
    const { mean, variance } = tf.moments(numbers, 0)
    numbers.sub(mean).div(variance.pow(.5))
    ```
    * 59. Applying Standardization
    * 60. Debugging Calculations
    * 61. What Now?
    * 62. Linear Regression
    * 63. Why Linear Regression?
    * 64. Understanding Gradient Descent
    * 65. Guessing Coefficients with MSE
    * 66. Observations Around MSE
    * 67. Derivatives!
    * 68. Gradient Descent in Action
    * 69. Quick Breather and Review
    * 70. Why a Learning Rate?
    * 71. Answering Common Questions
    * 72. Gradient Descent with Multiple Terms
    * 73. Multiple Terms in Action
## Section 6: Gradient Descent with Tensorflow
* 74. Project Overview
* 75. Data Loading
* 76. Default Algorithm Options
* 77. Formulating the Training Loop
* 78. Initial Gradient Descent Implementation
* 79. Calculating MSE Slopes
* 80. Updating Coefficients
* 81. Interpreting Results
* 82. Matrix Multiplication
* 83. More on Matrix Multiplication
* 84. Matrix Form of Slope Equations
* 85. Simplification with Matrix Multiplication
* 86. How it All Works Together!
## Section 7: Increasing Performance with Vectorized Solutions
* 87. Refactoring the Linear Regression Class
* 88. Refactoring to One Equation
* 89. A Few More Changes
* 90. Same Results? Or Not?
* 91. Calculating Model Accuracy
* 92. Implementing Coefficient of Determination
* 93. Dealing with Bad Accuracy
* 94. Reminder on Standardization
* 95. Data Processing in a Helper Method
* 96. Reapplying Standardization
* 97. Fixing Standardization Issues
* 98. Massaging Learning Rates
* 99. Moving Towards Multivariate Regression
* 100. Refactoring for Multivariate Analysis
* 101. Learning Rate Optimization
* 102. Recording MSE History
* 103. Updating Learning Rate
## Section 8: Plotting Data with Javascript
* 104. Observing Changing Learning Rate and MSE
* 105. Plotting MSE Values
* 106. Plotting MSE History against B Values
## Section 9: Gradient Descent Alterations
* 107. Batch and Stochastic Gradient Descent
* 108. Refactoring Towards Batch Gradient Descent
* 109. Determining Batch Size and Quantity
* 110. Iterating Over Batches
* 111. Evaluating Batch Gradient Descent Results
* 112. Making Predictions with the Model
## Section 10: Natural Binary Classification
* 113. Introducing Logistic Regression
* 114. Logistic Regression in Action
* 115. Bad Equation Fits
* 116. The Sigmoid Equation
* 117. Decision Boundaries
* 118. Changes for Logistic Regression
* 119. Project Setup for Logistic Regression
* 120. Project Download
* 121. Importing Vehicle Data
* 122. Encoding Label Values
* 123. Updating Linear Regression for Logistic Regression
* 124. The Sigmoid Equation with Logistic Regression
* 125. A Touch More Refactoring
* 126. Gauging Classification Accuracy
* 127. Implementing a Test Function
* 128. Variable Decision Boundaries
* 129. Mean Squared Error vs Cross Entropy
* 130. Refactoring with Cross Entropy
* 131. Finishing the Cost Refactor
* 132. Plotting Changing Cost History
## Section 11: Multi-Value Classification
* 133. Multinominal Logistic Regression
* 134. A Smart Refactor to Multinominal Analysis
* 135. A Smarter Refactor!
* 136. A Single Instance Approach
* 137. Refactoring to Multi-Column Weights
* 138. A Problem to Test Multinominal Classification
* 139. Classifying Continuous Values
* 140. Training a Multinominal Model
* 141. Marginal vs Conditional Probability
* 142. Sigmoid vs Softmax
* 143. Refactoring Sigmoid to Softmax
* 144. Implementing Accuracy Gauges
* 145. Calculating Accuracy
## Section 12: Image Recognition In Action
* 146. Handwriting Recognition
* 147. Greyscale Values
* 148. Many Features
* 149. Flattening Image Data
* 150. Encoding Label Values
* 151. Implementing an Accuracy Gauge
* 152. Unchanging Accuracy
* debug node --inspect-brk .\index_num.js 
* 153. Debugging the Calculation Process
* 154. Dealing with Zero Variances
* 155. Backfilling Variance
* node --max-old-space-size=4096 index_num.js
* 156. Handing Large Datasets
* node --inspct-brk memory.js
* 157. Minimizing Memory Usage
* 158. Creating Memory Snapshots
* 159. The Javascript Garbage Collector
* 160. Shallow vs Retained Memory Usage
* 161. Measuring Memory Usage
* 162. Releasing References
* 163. Measuring Footprint Reduction
* 164. Optimization Tensorflow Memory Usage
* 165. Tensorflow's Eager Memory Usage
* 166. Cleaning up Tensors with Tidy
* 167. Implementing TF Tidy
* 168. Tidying the Training Loop
* 169. Measuring Reduced Memory Usage
* 170. One More Optimization
* 171. Final Memory Report
* 172. Plotting Cost History
* 173. NaN in Cost History
* 174. Fixing Cost History
* 175. Massaging Learning Parameters
* 176. Improving Model Accuracy
* 177. Loading CSV Files
* 178. A Test Dataset
* 179. Reading Files from Disk
* 180. Splitting into Columns
* 181. Dropping Trailing Columns
* 182. Parsing Number Values
* 183. Custom Value Parsing
* 184. Extracting Data Columns
* 185. Shuffling Data via Seed Phrase
* 186. Splitting Test and Training