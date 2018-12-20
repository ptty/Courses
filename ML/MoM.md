# Week 1

  ### ML: 

  Arthur Samuel (1959) - chess playe - ability to learn how to play w/ o programming it explicitly.
  	

  Tom Michael (1998):
  	Problem defintion: A copoture program is said to leatn from experience E with respect to some task and some performance measuire P if its perofrmance on T as measured by P improves wwith 	experience E. i.e. performance imprvoes with more experience on T.

   		- E - expreience to play 
   		- T - play chess
   		- P - wins over chess player

  Spam:

   			- T-  classify email as spam or not spam
   			- E - label emails as spam or not spame 
   			- P - the number of emails correctly identified as spam



  			Supervise learning
  			Unsupervice learninig

  ### Supervised learning
    wE always tell what is currect answer during teaching.

    Regression - predict how much it will be in future

    	Problem: Large inventory. You want of identical items. You want to predict how many of there items will sell ove the next 3 month;

    Classiifcation  -defined number of results

    	Problem:
    		Softeware examine individual customer accounts and for each account decide if it has been hacked/compromised

       ### Example 1:
    		  	PLot : Y - Price / Size in feet - X

    		  	Instead straight line plot curve

    		  	Supervice learning - give  algorithm a right perices to calculte the right one

  ![](img/ScreenShot_2018-11-11_at_14.59.47.png)

  ### Example 2:
        Is a tumor malignant(no cancer) or benign based on size?

        Breast cancer
        	X - size of tumore
        	Y - Malignant[0, 1]?

        	Yes/No plot

        Classification problem (Malignent / benign)

        1 feature is used - tumor size 

  ![](img/ScreenShot_2018-11-11_at_15.02.54.png)

  ### Example 3:
  		    Is a tumor benign based on size & age?

  ![](img/ScreenShot_2018-11-11_at_15.05.37.png)

  		    Many (finite) feature, how to do with it?

  ### Unsupervised learning

  Unsupervised learning has not lables on the data facts. The goal is to identify clusters (clustering algoritm). Example; google news, genese.

  Other examples:
  	-Social network analysis
  	-Market segement ()
  	-Astronomical analysis( how galazies are formed)
  	-Cocktail party problem (2 speakers, split speakers as result)

  ### Linear Regression
  	Context: houses perices

  	Notaion:
  		- m = number of training examnples
  		- x = inpit vraiable / feature
  		- y = out variable / target

   (x,y) - one trainig examples
   (x(i), y(i)) - i-th training examples

   ![](img/ScreenShot_2018-11-11_at_18.16.07.png)

  Training set - > Learining argoritm -> h 

  ![](img/SNAG_Program-0000.png)

  Size of house (x) -> h -> Estimated price (y)

  ![](img/ScreenShot_2018-11-11_at_18.18.55.png)

  #### How to choose Tetha0, Theta1?

  Idea: Choose Theta0, Theta1 so that h(x) is close to y for our training example (x,y)

  Cost function (squered error)  - The expression idea, find parameters Theta0, theta1 where h(x) -y is min

  ![](img/ScreenShot_2018-11-11_at_18.24.50.png)

  #### Cost function

  ![](img/ScreenShot_2018-11-11_at_18.26.16.png)

  Hyposis function h(x)
  Cost function J(Theta)

  Corelations
  ![](img/ScreenShot_2018-11-11_at_18.28.39.png)
  ![](img/ScreenShot_2018-11-11_at_18.29.39.png)
  ![](img/ScreenShot_2018-11-11_at_18.30.48.png)
  ![](img/SNAG_Program-0001.png)
  ![](img/SNAG_Program-0003.png)



  #### Cost function intuition (Contour plots)

  More parametrs leads to more dimentions to cost function, 2 dim - can be shown as below:

  3d surface plot (Convex function - bowl shape)

  ![](img/ScreenShot_2018-11-11_at_18.32.54.png)

  Contor plots  (instead 3d)

  ![](img/ScreenShot_2018-11-11_at_18.34.59.png)
  ![](img/ScreenShot_2018-11-11_at_18.35.39.png)

  ![](img/SNAG_Program-0004.png)
  ![](img/SNAG_Program-0005.png)

  ### Parameter Learning
  #### Gradient descent
   Problem setup:
      having some function J(Theta0, Theta1)
      want to min J(Theta0, Theta1)

   Outline:
   	Start with some Theta0, Theta1
   	Keep changing Theta0, Theta1 to reduce J(Theta0, Theta1) until end up at minimum

  ![](img/SNAG_Program-0005.png)

  Definition of algorithm:
  ![](img/SNAG_Program-0006.png)

  -Alpha - steps size in gradient (learning rate)
  -Partial derivative of J(Theta0, Theta1) - determins direction

  #### Gradient descent intuition

  ![](img/SNAG_Program-0007.png)

  Alpha too small:
  	- lots of steps in finding min

  Alpha too large:
  	- can overshoot the minimum (fail to converge, even diverge)

  If we are in local min - Thata1 ==Theta1	

  If Alpha is fixed, derivitive will have small and snaller steps when it's approaching minimum. No need to decrease Alpha.

  ![](img/SNAG_Program-0008.png)
  ![](img/SNAG_Program-0009.png)
  ![](img/SNAG_Program-0010.png)

  ### Gradient descent + linear regression
  sometimes call as "Batch" gradient descent - each step of gradient descvent uses all the training examples.

  ![](img/SNAG_Program-0011.png)
  ![](img/SNAG_Program-0012.png)
  ![](img/SNAG_Program-0013.png)
  ![](img/SNAG_Program-0015.png)
  ![](img/SNAG_Program-0014.png)


  ### Matrix and Vectors
  #### Matrix
  	rows x columns; 

  	R(4x2)
  	Matrix element - A(i,j); A(1,2) = 1 row and 2 column;
  	upper case - matrix ref.

  #### Vector
     1 column; R(4) - for dim vector;
     y(2); (lower case - vectors ref)

  #### Scalar
  	y = 2; not vector, not matrix;

  ### Matrix Addition
  #### Same dim
  	 3x2 + 3x2 = 3x2
  	 3x2 + 2x2 = error

  #### Scalar Multiplication
  	2 x [3x2] = [3x2]
  	4 / [3x2] = 1/4*[3x2] = [3x2]

  ![](img/SNAG_Program-0016.png)

  ### Multiplication
  ####Matrix - Vector
       3x2 * 2x1 = 3x1

  ![](img/SNAG_Program-0018.png)

  How to calculate Hyposis function using trianing set:
  ![](img/SNAG_Program-0017.png)

  #### Matrix - Matrix
  	2x3 * 3x2 = 2x2

  ![](img/SNAG_Program-0020.png)

  ![](img/SNAG_Program-0019.png)

  #### Matrix Multiplication Properties
  	3*5 = 4*3 - commutative
  	
  	A*B != B*A - not commutative

  	3*5*2 = 5*2*3 - associative

  A*B*C = A*(B*C) = (A*B)*C  - associative

  ##### Identity Matrix
     A*I = I*A = A

     mXn * nXn = mXm * m*n  = mXn - commutative

  ![](img/SNAG_Program-0021.png)


  #### Matrix inverse and transpose

  ##### Inverse
   1 = identity
   3*(3^-1) = 1
  0*(0^-1) = undefined

   Matrix inverse
   AA(A^-1) = A^-1*A = I

  A = [3,4;2,6]
  p[inv](A) = [0.4 , -0.1; -0.05, 0,075] - inversed A

  A = [0,0; 0,0] - non-inversable
  ##### TRanspose


  A = [1,2,0; 3,5,9]
  A' = [1,3; 2,5;0,9] - transposed

  ![](img/SNAG_Program-0022.png)

# Week 2

  ## Multivariate Linear Regression

  ### Mutiple features
    Context: Predict house price with more features (size, number of beds, no floors, age, price)

    x1 - size
    x2 - bedrooms
    x3 - floors
    x4 - age
    y = price

  ![](img/SNAG_Program-0027.png)
  ![](img/SNAG_Program-0023.png)
  ![](img/SNAG_Program-0024.png)
  ![](img/SNAG_Program-0025.png)
  ![](img/SNAG_Program-0026.png)

  ### Gradient Descent in Practice I - Feature Scaling

  Contour is skinny if the scale of feature is very different, but scaling it down it allows contour plots to be more effective for gradient descent to work out the minimum (w/ less steps)

  Idea: get every feature into approx. a -1 <= x(i) <= +1 range;

  ![](img/SNAG_Program-0029.png)

  #### Mean normalization
  	x(i) = (size(i) - mean(i)) / range(i) - ( max - min)

  ![](img/2018-11-15_22-03-12.339.png)

  ### Gradient Descent in Practice II - Learning Rate

  ![](img/SNAG_Program-0030.png)
  ![](img/SNAG_Program-0031.png)

  ### Features and Polynomial Regression

  Context: House price prediction with frontage and depth as features
  Idea: 
  	you can create 1 feature instead using 2 features
  	area = frontage * depth;
  ####  Polynomial Regression

  Options:
  h(x) = Theta0+Thata1*x+Theta2*x^2 (dependes on training set)
  h(x) = Theta0+Thata1*x+Theta2*x^2+Theta3*x^3
  h(x) = Theta0+Thata1*x+Theta2*sqrt(x)

  ![](img/SNAG_Program-0032.png)


  ## Computing Parameters Analytically
  ### Normal Equation
  	alternative to Gradient Descent to get 0

  	Intuition:
  		Assume: Thatea is a raw number

  		J(Thata) = a*Thata^2 + b*Theta + c

  		Solution: Get J derivative and assign it to 0

  ![](img/SNAG_Program-0028.png)
  ![](img/SNAG_Program-0033.png)

  Feature scaling is NOT required for Normal Equation!

  ![](img/SNAG_Program-0034.png)

  Pros & Cons:

  ![](img/SNAG_Program-0035.png)


  ### Normal Equation Noninvertibility
  Sometimes X'*X non-invertable.

  ![](img/SNAG_Program-0036.png)

  ## Octave
  ### Operations
     - not equal - ~=
     - PS('>> ');  - change command line string
     - ; in the end supresses the output
     - b = 'hi'
     - disp(sprintf('2 deciamls: %0.2f', a))
     - format long / format short]

  ### Matrixes

    - v = 1:0.1:2 
    - 2*ones(2,3)  - 2x3
    - zeros(1,3) - 1x3
    - rand(3,3) 3x3
    - randn(1,3) - gausian dist
    - hist(w,50) - print histogram w/ 50 bars
    - help eye
    - size(A)
    - length(A) - longer dimention
    - A(2, :) - show everything second row
    - A(:, 2) - show everything in second column
    - A([1 3], :) - get from 1 and 3 row 
    - A(:, 2) = [ 10; 11; 12] - replace second column with new values
    - A(A, [100; 101; 202]) - add new column to right
    - A(:) - put all elements into vector
    - C = [A B] or [A, B] - conctatination 2 matrcies
    - C = [A; B] - put B matrix on bottom of A
    - 

  ### Data operations

  - pwd - path
  - ls - files list
  - cd - go to dir
  - load('file.dat') - load file into Octave
  - who - what variables in mem
  - clear A - remove variable from mem
  - v = priceY(1:10) - First 10 elements 
  - save hello.dat v; - save data into file
  - save hello.dat v - ascii; - text file save

  ### Computation
   - A*C - matrix multiplaication
   - A .* B - element-wise multiplication 
   - A .^ B 
   - 1 ./A
   - log(v)
   - exp(v)
   - abs(v)
   - -v
   - v+ones(length(v), 1) - increase everything by 1
   - v + 1
   - A' - transpose
   - [val, ind] = max(A) - return max and location
   - a < 3 - return matrix 0,1
   - find(a < 3) - return idx of element meets the criteria
   - A = magic(3) - generates 3x3
   - [r,c] = find(A>=7) - return row and col numbers
   - sum()
   - floor()
   - ceil()
   - rand(3) - 3x3
   - max(A, [], 1) - max colum wise
   - max(A, [], 1) - max row wise
   - sum(A, 1) - sum colum wise
   - sum(A, 2) - sum row wise
   - flipup(A)
   - pinv(A) - inverse
   
  ### Plotting

    - plot(t, y1); t=[0, 0.1, 2]; y1=sin(2*pi*4*t);
    - hold on; - plots on top of existing
    - plot(t, y1, 'r') - plots red
    - xlable('time');
    - title('miplot')
    - print -dpng 'myplot.png'
    - help plot
    - close - close plot
    - figure(1); plot(t,y1)'
    - figure(2); plot(t, y2);
    - subplot(1,2,1);
    -     plot(t, y2);
    -     subplot(1,2,2);
    -     plot(t, y1);
    - axis([0.5 1 -1 1]);
    - clf; - clear figure
    - imagesc(A); - grid of color
    - imagesc(A), colorbar, colormap gray; - 3 commands runs each after another


  ### Control Statements: for, while, if statement
   
  #### FOR
   - for i=1:10, v(i) = 2^i; end;
   - indices=1:10;
      -for i=indices, disp(i); end;

  #### WHILE

  - i =1;
  - while i <= 5,
      + v(i) = 100;
      + i = i+ 1;
      + end;
  - i=1;


  - while true,
      + v(i) = 999;
      + i = i+1;
      + if i == 6,
          * break;
      + end;
     end;

  #### IF

  - if (v1) == 1,
  -   disp(' one');
  - elseif v(1) == 2,
  -   disp('two')
  - else
  -   disp('other');
  - end;   

  #### Functions
    1. create file myFunction.m


    2. add body:
    
      function y = myFunction(x)
      y= x^2;
   
   3. cd to location
   4.myFunction(5); 

  addpath('path to functions files');

  ##### Functions (many return values)
  function [y1,y2] = myFunction(x)  -- multi values returned
      y1= x^2;
      y2= x^3;

  ##### Vectorization 

  ![](img/2018-11-15_11-29-40.696.png)
  ![](img/2018-11-15_11-36-57.054.png)

# Week 3
  ## Classification (binary classification problem)
  	called Logistic Regression 

  ####  Problem:
  		Email: Spam/Not Spam
  		Online Transactions: Fraudulent (Yes, No)
  		Tumor: Mlignant/Benign

  		y e {0,1} - binary classification problem

  		0 - Negative class
  		1 - Positive class

  		h = O^T*x

  		Threshold: 
  			if h >= 0.5 - predict y = 1;
  			if h < 0.5 - predict y = 0;

  ####  Why linear regression does not fit to classification problem?
  		1. If there is a fact far behind most of the training set facts, it affects the performance badly		
  		2. h can be > 1 or < 0

  ![](img/SNAG_Program-0037.png)

  #### What is solution?
  		Logistic Regression (classification algoritm, not regression): 0 <= h <= 1

  ## Hypothesis Representation
  	Goal:  0 <= h <= 1

  	h = g(Theta^T*x)
  	g - sigmoid, or logisitc function 


  ![](img/SNAG_Program-0038.png)

  ##### Interpretation
  	h  becomes probabilty P - 1 - 100% probable
  ![](img/SNAG_Program-0039.png)

  ## Decision boundary
  	y = 1 if h >= 0.5 or  Theta^T*x >= 0 
  	y = 0 if h < 0.5 or  Theta^T*x < 0 

  ![](img/SNAG_Program-0040.png)
  ![](img/SNAG_Program-0041.png)
  NOTE: decision boundary is a property of the hyposis, not the data set.

  ### Non-liner decision boundaries
  	h = g(Theta0 + Theta1*x1 + Theta2*x2 + Theta3*x1^2 + Theta4*x2^2);

  ![](img/SNAG_Program-0042.png)

  ## Logistic regresion. Cost Function
  	Hot to choose paramter Theta?

  ![](img/SNAG_Program-0043.png) 

   the problem:
   	the sigmoid / logistic functio leads is non-convex Cost function (J)(having many minimums) so it's hard to find global minimum	

  ![](img/SNAG_Program-0044.png) 

  solution:
  	Different cost function:
  		J(h(x), y) = { -log(h(x))} if y = 1; 
  		               -log(1-h(x))} if y = 0; }

  ![](img/SNAG_Program-0045.png) 		               
     Intuition:
         if h(x) = 0 (predict P(y=1) = 0), but y = 1 we penalize learning argorthm by very large cost;

  ![](img/SNAG_Program-0046.png)         
  Intuition:
         if h(x) = 1 (predict P(y=0) = 0), but y = 0 we penalize learning argorthm by very large cost;


  ![](img/SNAG_Program-0047.png)   

  ## Simplified Cost Function and Gradient Descent

  ![](img/SNAG_Program-0048.png)  
  ![](img/SNAG_Program-0050.png)  
  ![](img/SNAG_Program-0051.png)  
  ![](img/SNAG_Program-0052.png)  

  ## Advanced Optimization
      Optimized algorithms:
          - Conjugate gradient
          - BFGS
          - L-BFGS

      Advantages:
          - No need to manually poick Alpha
          - Often faster than gradient descent
      Disadvanteges:
          - More complex


  ![](img/SNAG_Program-0053.png)  
  ![](img/SNAG_Program-0054.png)  

  ## Multiclass Classification: One-vs-all
      Examples: 
  ![](img/SNAG_Program-0055.png)  
  ![](img/SNAG_Program-0056.png) 
  ![](img/SNAG_Program-0057.png) 

  ## Overfitting

  Linear regression:

  ![](img/SNAG_Program-0058.png) 

  Logistic regression:

  ![](img/SNAG_Program-0059.png) 

  ### How to address?
    
      1. We can plot and see (does not work for many feature as hard to plot)
      2. Reduce number of features
      3. Model select algorithm (reducing features by algorithm)
      4. Regularization (Keep all features but reduce magnitude/values of Theta)

  ## Regularization

    Intuition:

  ![](img/SNAG_Program-0060.png) 

    Small values of parameters Thata:
      - Simpler hyposiss (That3,4 ~ 0)
      - Less prone to overfitting


    Housing:
      -Features: x1,...,X100
      -Parameter: Theta0,...,Theta100  

  ![](img/SNAG_Program-0061.png) 

  Lambda - reg. parameter.

  Goal:
    Find tradeoff between: 
    - we would like to fit to traing set well
    - keeping hyposis relatively simple

    Usually Theta0 is NOT regularized.

  ![](img/SNAG_Program-0062.png) 
  ![](img/SNAG_Program-0063.png) 

  ## Regularized Linear Regression

  ### Reg. for Gradient Descent

  ![](img/SNAG_Program-0064.png) 
  ![](img/SNAG_Program-0065.png) 

  ### Reg. for Normal Equasion

  ![](img/SNAG_Program-0066.png) 

  L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including x_0x 
  0
  ​  ), multiplied with a single real number λ.

  Recall that if m < n, then X^TXX 
  T
   X is non-invertible. However, when we add the term λ⋅L, then X^TXX 
  T
   X + λ⋅L becomes invertible.

  X is non-invertible. However, when we add the term λ⋅L, then 


  ## Regularized Logistic Regression

  ![](img/SNAG_Program-0067.png) 
  ![](img/SNAG_Program-0068.png) 

# Week 4 (Neural networks)


##Non-linear Hypotheses

Why do we need it NN?
 1. It might be hard to find non-linear function due to 
 - many features 
 - hard to calculate
 - Theta(n^3)
 2. Recognize if the image is car or not?
 - 50x50 pixel -> 2500 pixel; n = 2500 (7500 if RGB)
 - Quadratic features = 3 mil
 
## Neurons and the Brain

![](img/SNAG_Program-0069.png) 
![](img/SNAG_Program-0070.png) 

## Model Representation 1

![](img/SNAG_Program-0072.png) 
![](img/SNAG_Program-0073.png) 


### Neuron 
  Key terms:
    1. Neuron input wises
    2. Neuron output
    3. Bias unit
    4. Weights = parameters *theta*

![](img/SNAG_Program-0075.png) 

### Neuron Network
  Key terms:
  - 1 Layer - Input layer
  - Last Layer - Output Layer 
  - Layers in middle = hidden layers
  - Units - elements of layer

![](img/SNAG_Program-0076.png) 

![](img/SNAG_Program-0077.png) 

## Model Representation 2
Forward propogatio method;

1. Definig z vector;
2. x = a(1) vecotr -> z(2) = Thata(1)*a(1)
3. a0(2) = 1

![](img/SNAG_Program-0078.png) 

Simplified version of the NN is Logistic Regression with one different it uses a vector instead of x, so it a "learns" from x vector

![](img/SNAG_Program-0079.png) 

other NN architectectures:

![](img/SNAG_Program-0080.png) 


## Examples and Intuitions I

### non-linear classification example: XOR/XNOR

i.e. both x1, x2 is true or both are false i.e. y = x1 AND x2
![](img/SNAG_Program-0081.png) 

#### Logical AND NN:

![](img/SNAG_Program-0082.png) 

#### Logical OR NN:
![](img/SNAG_Program-0083.png) 

## Examples and Intuitions II

### complex non-linear 
  
#### NOT x1 and Not X2

![](img/SNAG_Program-0084.png) 

#### x1 XNOR x2

combining:
   1. x1 AND x2; 
   2. (NOT x1) AND (NOT x2);
   3. x1 OR x2

1, 2 - layer 2
3 - layer 3


![](img/SNAG_Program-0085.png) 

## Multiclass Classification

Example: recognize 1-9 number from picture.

Problem (one vs all):
 Rcognize Pedestrial, Car, Motorcycle, Truck on the picture

Design solution:
    1. 4 output layer units (each for each case);
        1.1 when pedestrial: h = [1; 0; 0; 0]
        1.2 when car: h = [0; 1; 0; 0]
        ...

![](img/SNAG_Program-0086.png) 

y  represts vector rather than list of number 1,2,3,4

the goal is to find Thatas so h ~ y

![](img/SNAG_Program-0087.png) 


# Week 5

## Cost Function

   L - total number of layers
   s(l) = no of units (not counting bias unit) in layer l
   K - no of class
   S(L) = no of output units


Two type of classificaiton problem:
  1. Binary 
      y e {1,0}

      h = 1 unit
      S(L) = 1
      K = 1 (output units)

      2. Multi-class classification, say, 4 classes
        y e R^K e.g y = [1; 0; 0; 0]; [0; 1; 0; 0] ..
        h = R^K
        S(L) = K (K >= 3)

![](img/SNAG_Program-0088.png) 

### Cost function definition
  NOTE: regularization does NOT applied for bias units.

  SL - output unit number
  K - output unit in output layer = output classes
  K = No of classes - 1

  ![](img/SNAG_Program-0089.png) 
  ![](img/SNAG_Program-0090.png) 

NOTE: Regularization terms does not count bias unit

### Backpropogation
  i.e. how to minimize Cost gunction

![](img/SNAG_Program-0091.png) 

 Case 1: just x,y

![](img/SNAG_Program-0092.png) 

#### Backpropogation algorithm
How to minimize J(0)? i.e. calculation of derivative for NN

Steps:
  1. Forward propogation - allow to compute activation numbers for all neurons (a)
  2. In order to calculate derivatives , we use backward calculation algorithm:
   for each node we calculate error for every node delta.j(l)

error in output layer is a __delta__ between the value in learning set and calculated by activation unit

g'(z(3))  = a(3).*(1-a(3)) - g' is derivateive

Notes:
1. No __delta__ for input layer.
2. Ignoring Lambda for now
  
  
![](img/SNAG_Program-0093.png) 

DELTA - accumulate lower case delta 

![](img/SNAG_Program-0094.png) 

D.ij(l) is a partial derivative for J(0)




## Backpropogation intuition 
![](img/SNAG_Program-0095.png) 
![](img/SNAG_Program-0096.png) 


## Implementation Note: Unrolling Parameters

The problem:
  if fminunc function is used to find optTheta, it accepts vectors rather than matrices. How to deal with it? - unroll matrix to vector

![](img/SNAG_Program-0097.png) 
![](img/SNAG_Program-0098.png) 

Steps:
1. Having init parameters Thata1, Theta 2, Theta 3
2. Unroll to get initialTheta to pass too
3. fminunc(@costFunction, initialTheta, options)
4. Define costFunction
4.1. in costFunction, from thetaVec, get Theta1, Theta2, Theta3
4.2. in costFunction, use forward prop/back to compute D(1), D(2), D(3) and J(Theta)
4.3. unroll D(1), D(2), D(3) to get gradientVec


## Gradient Checking

Idea: test the implementaiton of prop & back prop is correct

Impl: to calculated derivative approx w/o actual derivative execution

![](img/screenshot2018-11-21at14.45.49.png)
![](img/screenshot2018-11-21at14.48.08.png)

__Implementation notes:__
![](img/screenshot2018-11-21at14.51.25.png)

## Random Initialization

last thing needed for NN training - initialTheta

What does it set to?

Say, initalTheta = 0 than:
a1(2) = a2(2); 
delta1(2) = delta2(2);
Theta01(1) = Theta02(1)
derv01(1) = deriv02(1)

![](img/screenshot2018-11-21at14.58.28.png)
![](img/screenshot2018-11-21at15.01.04.png)


## Putting It Together

Steps:
1. Choose the NN architecture, how
  1.1 No of input = dimentions of features
  1.2 No. output units = number of classes
  1.3. No of hidden = default is 1 hidden layer
  1.4 No of hidden units = the more the better
![](img/screenshot2018-11-21at15.07.34.png)
2. Training a neural network
![](img/screenshot2018-11-21at15.11.29.png)
3. Training a neural network
![](img/screenshot2018-11-21at15.13.23.png)
![](img/screenshot2018-11-21at15.17.40.png)

# Week 6

## Deciding What to Try Next

Suppose we have housing prices regularized linear regression but it gives unacceptable error when hypothesis applied for new data, what to do?

Options:
 1. More training examples
 2. Try smaller sets of features
 3. Try to get additional features
 4. Try to add plynomian features
 5. Try to descrease Lambda
 6. Try ito ncreate Lambda

There is technic gives a hint what will work - Machine learining diagnositc.

Diagnostic - a test you can run, to get insight into what is or isn't working with an algorithm, and which will often give you insight as to what are promising things to try to improve a learning algorithm's

Diagnostic may be time consuming but it safes time later;

## Evaluating a Hypothesis

Low training error does not mean it will be performant for new data.
It's hard to plot hypothsis due to number of features. 

What to do?
  1. Split data into 2 portion
  2. 70% -training set (randomly)
  3. 30% - testing set (m.test - no of test examples) (randomly)

### Learning procedure for linear regression
  1. Learn parameter Theta from training data (70%) (min training error J(Theta))
  2. Compute test set error (J(Theta)) (taking Theta from step 1) using test set.

![](img/screenshot2018-11-21at17.40.55.png)

### Learning procedure for logistic regresion
Misclassification error - another technic

![](img/screenshot2018-11-21at17.44.08.png)

## Model Selection and Train/Validation/Test Sets
Model selection 


  1. you're left to decide what degree of polynomial to fit to a data set.
  2.  suppose you'd like to choose the regularization parameter longer for learning algorithm

Trainig set Cost may NOT be reliable

switch data into what we discover is called the train, validation, and test sets and see the results for different models

Model selection is degree of polynomial.

Say, we choose 5th polinom as it gives best performance J.test(Theta(5)). 
However, there is problem: J.test(Theta(5)) is likely to be optiomistic estimates of generalization error i.e. (d = degree of polynomial) is fit to test set. 

![](img/screenshot2018-11-22at14.18.39.png)
Conclusion: The performance of particular degree of polinom is not always reliable as it overfitting the testing set.

![](screenshot2018-11-22at14.21.05.png)
![](screenshot2018-11-22at14.22.57.png)

How to select the right model?

![](screenshot2018-11-22at14.25.34.png)




























 







