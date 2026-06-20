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
