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

$$z = y_i^2 - 2w x_i y_i - 2by_i + w^2x_i^2 + 2w x_i b + b^2$$ <br>
Rearranging <br>
$$z = w^2x_i^2 + 2w x_i b +   b^2$$ - 2w x_i y_i - 2by_i + y_i^2   <br>
`z = 13.3505x^2 + 7.1971xy + y^2 - 119.6886x - 34.2857y + 303.7143`

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
Now plot this curve in Desmos OR any other 3D calculator <br>
https://www.desmos.com/3d <br>

You now know the minimum of the graph is around here, which means the loss is minimum around here. <br>


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
