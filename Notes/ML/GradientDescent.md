Find the linear regression model with the below dataset.
Essentially we have to find the miles per gallon, given the weight of the vehicle.
| Pounds in 1000s (feature) | Miles per gallon (label) | 
| --- | --- | 
|3.5	| 18 |
|3.69	|15|
|3.44|	18|
|3.43|	16|
|4.34	|15|
|4.42	|14|
|2.37|	24|



The formula for the model's prediction is:


$$\hat{y}_i = w \cdot x_i + b$$

The Mean Squared Error (MSE) loss for $n = 7$ data points is:


$$Loss = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{7} \sum_{i=1}^{7} (y_i - (w \cdot x_i + b))^2$$

As we know: **$(A - B)^2 = A^2 - 2AB + B^2$**.

Let's assign our variables to match the identity:

* $A = y_i$
* $B = (w \cdot x_i + b)$

Plugging our components into the expanded format yields:


$$(y_i)^2 - 2(y_i)(w \cdot x_i + b) + (w \cdot x_i + b)^2$$

Now, expand the final $(w \cdot x_i + b)^2$ term using the same binomial rule:


$$(w \cdot x_i + b)^2 = w^2x_i^2 + 2w x_i b + b^2$$

Combine them back together into one fully expanded, flat polynomial expression:

$$y_i^2 - 2w x_i y_i - 2by_i + w^2x_i^2 + 2w x_i b + b^2$$ <br>
OR <br>
$$z = y_i^2 - 2w x_i y_i - 2by_i + w^2x_i^2 + 2w x_i b + b^2$$ <br>

<b> Here,</b> <br>
$$x_i$$ = Feature, For a specific (ith) sample, the weight of the vehicle <br>
$$y_i$$ = Label, For a specific (ith) sample, the miles per gallon (efficiency) of the vehicle <br>
z = Loss <br>
w = weight for feature $$x_i$$ <br>
b = bias term. <br>



Here is the step-by-step algebraic expansion of your dataset to build the exact MSE loss equation and find its true minimum.

---

### 1. The Components from Your Dataset

Let $x$ be the weight parameter (**instead of w**), $y$ be the bias parameter (**instead of b**), and $z$ be the total Mean Squared Error (MSE) Loss. For $n = 7$ data points, the loss formula is:

$$z = \frac{1}{7} \sum_{i=1}^{7} (x \cdot X_i + y - Y_i)^2$$
$$z = y_i^2 - 2x x_i y_i - 2yy_i + x^2x_i^2 + 2x x_i b + y^2$$ <br>
$$z = \frac{1}{7} \sum_{i=1}^{7} y_i^2 - 2x \frac{1}{7} \sum_{i=1}^{7} x_i y_i - 2y\frac{1}{7} \sum_{i=1}^{7} y_i + x^2\frac{1}{7} \sum_{i=1}^{7} x_i^2 + 2x y \frac{1}{7} \sum_{i=1}^{7} x_i  + y^2$$ <br>



To expand this into a quadratic equation, we need five summations calculated directly from your table:

| $X_i$ (Pounds) | $Y_i$ (MPG) | $X_i^2$ | $X_i \cdot Y_i$ | $Y_i^2$ |
| --- | --- | --- | --- | --- |
| 3.50 | 18 | 12.2500 | 63.00 | 324 |
| 3.69 | 15 | 13.6161 | 55.35 | 225 |
| 3.44 | 18 | 11.8336 | 61.92 | 324 |
| 3.43 | 16 | 11.7649 | 54.88 | 256 |
| 4.34 | 15 | 18.8356 | 65.10 | 225 |
| 4.42 | 14 | 19.5364 | 61.88 | 196 |
| 2.37 | 24 | 5.6169 | 56.88 | 576 |
| **Sum ($\sum$): 25.19** | **Sum ($\sum$): 120** | **Sum ($\sum$): 93.4535** | **Sum ($\sum$): 418.91** | **Sum ($\sum$): 2126** |

---

### 2. Building the Coefficients

When you fully expand the squared term $\frac{1}{7}\sum(xX_i + y - Y_i)^2$, the coefficients are derived as follows:

* **Coefficient of $x^2$:** $\frac{\sum X_i^2}{7} = \frac{93.4535}{7} \approx \mathbf{13.3505}$
* **Coefficient of $xy$:** $\frac{2 \sum X_i}{7} = \frac{2 \cdot 25.19}{7} \approx \mathbf{7.1971}$
* **Coefficient of $y^2$:** $\frac{\sum 1}{7} = \frac{7}{7} = \mathbf{1.0}$
* **Coefficient of $x$:** $-\frac{2 \sum X_i Y_i}{7} = -\frac{2 \cdot 418.91}{7} \approx \mathbf{-119.6886}$
* **Coefficient of $y$:** $-\frac{2 \sum Y_i}{7} = -\frac{2 \cdot 120}{7} \approx \mathbf{-34.2857}$
* **Constant Term:** $\frac{\sum Y_i^2}{7} = \frac{2126}{7} \approx \mathbf{303.7143}$

$$z = y_i^2 - 2x x_i y_i - 2yy_i + x^2x_i^2 + 2x x_i y + y^2$$ <br>
$$z = \frac{1}{7} \sum_{i=1}^{7} y_i^2 - 2x \frac{1}{7} \sum_{i=1}^{7} x_i y_i - 2y\frac{1}{7} \sum_{i=1}^{7} y_i + x^2\frac{1}{7} \sum_{i=1}^{7} x_i^2 + 2x y \frac{1}{7} \sum_{i=1}^{7} x_i  + y^2$$ <br>
Rearranging <br>
$$z = x^2 \frac{1}{7} \sum_{i=1}^{7} x_i^2 + 2xy \frac{1}{7} \sum_{i=1}^{7} x_i  +   y^2 \frac{1}{7} \sum_{i=1}^{7} 1 - 2x \frac{1}{7} \sum_{i=1}^{7} x_i y_i - 2y\frac{1}{7} \sum_{i=1}^{7}y_i + \frac{1}{7} \sum_{i=1}^{7}y_i^2$$   <br>
`$$z = 13.3505x^2 + 7.1971xy + y^2 - 119.6886x - 34.2857y + 303.7143$$`

---

### 3. Your Final Desmos Equations

Copy and paste these exact equations into Desmos 3D to see the flawless alignment:

**Line 1 (The 3D Loss Bowl):**
`z = 13.3505x^2 + 7.1971xy + y^2 - 119.6886x - 34.2857y + 303.7143`

**Line 2 (The True Minimum Coordinate):**
`(-4.569, 33.585, 1.470)`

> **The True Minimum:** At its lowest point, the optimal weight is **$-4.569$**, the optimal bias is **$33.585$**, and the absolute minimum MSE loss value ($z$) is **$1.470$**.
>
> 
====


## Understanding and Solving the Partial Derivatives

To minimize this loss function using Gradient Descent, the model needs to know which direction is "downhill." We find this by taking the **partial derivative** of the loss function with respect to $x$ (weight) and $y$ (bias) independently.

When taking a partial derivative with respect to one variable, you treat the other variable as a constant.

Using the power rule ($\frac{d}{dx}[cx^n] = ncx^{n-1}$), let's differentiate your final Desmos loss equation:

### 1. Partial Derivative with respect to Weight ($x$)

$$\frac{\partial z}{\partial x} = \frac{\partial}{\partial x}(13.3505x^2 + 7.1971xy + y^2 - 119.6886x - 34.2857y + 303.7143)$$

* Derivative of $13.3505x^2$ is $2 \times 13.3505x = 26.701x$
* Derivative of $7.1971xy$ is $7.1971y$ (treating $y$ as a constant)
* Derivative of $y^2$, $-34.2857y$, and $303.7143$ are all $0$ (no $x$ present)
* Derivative of $-119.6886x$ is $-119.6886$

$$\frac{\partial z}{\partial x} = 26.701x + 7.1971y - 119.6886$$

### 2. Partial Derivative with respect to Bias ($y$)

$$\frac{\partial z}{\partial y} = \frac{\partial}{\partial y}(13.3505x^2 + 7.1971xy + y^2 - 119.6886x - 34.2857y + 303.7143)$$

* Derivative of $y^2$ is $2y$
* Derivative of $7.1971xy$ is $7.1971x$ (treating $x$ as a constant)
* Derivative of $-34.2857y$ is $-34.2857$
* All terms without $y$ become $0$

$$\frac{\partial z}{\partial y} = 7.1971x + 2y - -34.2857$$

---

## Solving for the Minimum Analytically

Before running gradient descent, we can prove where the exact minimum sits by setting both partial derivatives to zero (where the slope is perfectly flat at the bottom of the bowl):

1. $26.701x + 7.1971y = 119.6886$
2. $7.1971x + 2y = 34.2857 \rightarrow y = \frac{34.2857 - 7.1971x}{2}$

Substituting $y$ into the first equation:


$$26.701x + 7.1971\left(\frac{34.2857 - 7.1971x}{2}\right) = 119.6886$$

Solving this system of linear equations yields our exact target coordinate:

* **$x$ (Weight) $= -4.569$**
* **$y$ (Bias) $= 33.585$**

---

## Solving the Loss with Gradient Descent

Gradient Descent doesn't look at the whole system at once. Instead, it starts at an initial guess (usually $x=0, y=0$) and takes small steps down the hill using this iterative update rule:

$$\text{New Parameter} = \text{Old Parameter} - (\text{Learning Rate} \times \text{Slope})$$

Let's simulate **Step 1** using a conservative learning rate ($\alpha = 0.05$):

### Iteration 1

* **Start Position:** $x_0 = 0$, $y_0 = 0$
* **Calculate Slopes at this position:**

$$\frac{\partial z}{\partial x} = 26.701(0) + 7.1971(0) - 119.6886 = -119.6886$$


$$\frac{\partial z}{\partial y} = 7.1971(0) + 2(0) - 34.2857 = -34.2857$$


* **Update Parameters:**

$$x_{\text{new}} = 0 - (0.05 \times -119.6886) = \mathbf{5.984}$$


$$y_{\text{new}} = 0 - (0.05 \times -34.2857) = \mathbf{1.714}$$



### Iteration 2

Now, plug $x = 5.984$ and $y = 1.714$ back into your partial derivative formulas to calculate the next step's slopes, and repeat.

Over dozens of iterations, the step sizes will naturally shrink as the calculated slopes get closer to $0$, eventually landing right at the true minimum coordinate: **$(-4.569, 33.585)$**.

---

## The Final Linear Regression Model

Using these calculated parameters, your completed predictive model for vehicle fuel efficiency is:

$$\text{Miles Per Gallon} = -4.569 \times (\text{Pounds in 1000s}) + 33.585$$


# If we can find minima by setting the derivative to zero, then why we need gradient descent

That is the ultimate "million-dollar question" in machine learning. If we can just use high school calculus to set the derivatives to zero and solve the system of equations directly (which is called finding the **analytical or closed-form solution**), why do we bother with the slow, iterative process of Gradient Descent?

The short answer is: **scalability and mathematical limits.** While setting the derivative to zero works perfectly for simple 2D or 3D linear regression, it completely falls apart when dealing with real-world machine learning and deep learning. Here is exactly why we need Gradient Descent:

---

### 1. The "Big Data" Problem (Computational Complexity)

To solve a linear regression problem analytically with $k$ features, computers must perform matrix inversion. The computational cost of multiplying and inverting a matrix scales at roughly $\mathcal{O}(k^3)$.

* For your dataset with **1 feature** (vehicle weight), the computer has to invert a $2 \times 2$ matrix. It takes microseconds.
* If you want to predict house prices using **100 features**, the computer inverts a $100 \times 100$ matrix. Still very easy.
* What if you are training a modern Large Language Model (LLM) or deep neural network with **100,000 features** or billions of parameters? Inverting a matrix that size would require more computer memory and processing power than exists on earth.

**Gradient Descent bypasses this** because it never tries to look at the whole puzzle at once. It just asks: *"At my current position, which tiny step is downhill?"* It is incredibly cheap computationally and scales beautifully to massive datasets.

---

### 2. Non-Linear and Non-Convex Functions

The MSE loss function you just mapped out is a **convex function**—it forms a perfect, predictable bowl with exactly one bottom (global minimum).

Neural networks and complex algorithms have **non-convex** loss surfaces. They look like a chaotic mountain range with hundreds of peaks, fake valleys (**local minima**), and flat zones (**saddle points**).

When you try to set the derivative to zero on a non-convex function:

* The algebra often becomes **symbolically impossible** to solve (there is no algebraic formula to isolate the variables).
* Even if you could solve it, setting the derivative to zero would give you a massive list of *every single peak, local valley, and flat spot* on the map, with no easy way to know which one is the absolute lowest point.

---

### 3. Streaming Data (Online Learning)

An analytical solution requires you to have all your data sitting in a table at the exact same time to build your equations.

If you have a streaming system—like credit card transactions coming in live every second, or a self-driving car continuously learning from new road frames—you cannot re-run a massive matrix inversion every time a new data point arrives. Gradient Descent allows a model to simply take a few steps to adjust its weights incrementally as new data flows in.

---

### Summary: The Tool for the Job

| Feature | Setting Derivative to Zero (Analytical) | Gradient Descent (Iterative) |
| --- | --- | --- |
| **How it works** | Jumps straight to the bottom in one massive mathematical leap. | Takes small, blind steps downhill until it can't go any lower. |
| **When to use it** | Small datasets, simple linear models. | Deep learning, massive datasets, complex neural networks. |
| **Limitation** | Destroys computer memory as data grows; fails completely on complex shapes. | Requires tuning parameters (like learning rate) and takes multiple steps. |
---












---

## The Core Concept: How It Works

To understand gradient descent, you need to understand three key components:

* **The Cost Function (The Mountain):** This is a mathematical formula that measures how wrong your model's predictions are. The goal of gradient descent is to get this error as close to zero as possible.
* **The Gradient (The Slope):** In calculus, the gradient is the derivative of the cost function. It tells us the slope of the hill at our current position. If the slope is positive, we move backward; if it's negative, we move forward.
* **The Learning Rate (The Step Size):** Often denoted as $\alpha$ (alpha), this dictates how big of a step we take down the hill.

### The Mathematical Step

Mathematically, the algorithm updates the weights ($w$) of a model using the following formula:

$$w_{new} = w_{old} - \alpha \cdot \nabla L(w)$$

Where:

* $w$ represents the model's weights or parameters.
* $\alpha$ is the **learning rate**.
* $\nabla L(w)$ is the **gradient** (the direction of steepest ascent of the loss function $L$). We subtract it because we want to go *down*, not up.

---

## The Importance of the Learning Rate ($\alpha$)

Choosing the right learning rate is crucial. Think of it as your stride length down the mountain:

* **Too Small:** You take tiny, baby steps. The algorithm will be incredibly precise, but it will take a massive amount of time and computing power to reach the bottom.
* **Too Large:** You take giant, aggressive leaps. You might completely overshoot the bottom of the valley and end up climbing up the opposite hill, causing the algorithm to fail (diverge).

[Image comparing small learning rate versus large learning rate in gradient descent]

---

## 3 Main Types of Gradient Descent

Depending on how much data we look at before making a step, gradient descent comes in three flavors:

| Type | How it Works | Pros | Cons |
| --- | --- | --- | --- |
| **Batch Gradient Descent** | Calculates the error for the **entire dataset** before making one single update. | Stable convergence; smooth trajectory. | Very slow and memory-intensive for large datasets. |
| **Stochastic Gradient Descent (SGD)** | Updates the weights after looking at **each individual** data point. | Very fast; can escape local minima easily due to its erratic nature. | The path to the bottom is highly erratic and noisy; never truly settles. |
| **Mini-Batch Gradient Descent** | Strikes a balance by updating weights based on small groups (e.g., 32, 64, or 128 samples). | Fast, efficient, and takes advantage of hardware optimization. | Requires tuning an extra parameter (batch size). |

---

## The Ultimate Challenge: Local Minima vs. Global Minimum

If your cost function looks like a perfect bowl (convex), gradient descent is guaranteed to find the absolute lowest point (the **Global Minimum**).

However, real-world machine learning models often have complex, bumpy cost functions. The algorithm can get trapped in a **Local Minimum**—a dip that looks like the bottom to the algorithm, but isn't actually the lowest point on the entire mountain. Modern variants of gradient descent use "momentum" (like a rolling ball gathering speed) to help roll right past these shallow dips.

Would you like to explore how we mathematically calculate the gradient for a simple linear regression model, or perhaps look at how the learning rate is tuned in practice?
