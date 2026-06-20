Let's say we need to create a regression line for these data points.

| Feature (x) | Label (y) |
|---|---|
| 1 | 1.8 |
| 2 | 3.9 |
| 3 | 6.1 |
| 4 | 8.0 |
| 5 | 10.2 |

typically the equations shall have both weight and bias (but we are ignoring bias here).

**Loss (Mean Squared Error)**

The model prediction (no bias):

$$\hat{y}_i = w \cdot x_i$$

The error for each point is $\hat{y}_i - y_i$, so the MSE loss is:

$$Z = \frac{1}{n} \sum_{i=1}^{n} (w \cdot x_i - y_i)^2$$

Expanding with our 5 data points:

$$Z = \frac{1}{5} \left[ (w \cdot 1 - 1.8)^2 + (w \cdot 2 - 3.9)^2 + (w \cdot 3 - 6.1)^2 + (w \cdot 4 - 8.0)^2 + (w \cdot 5 - 10.2)^2 \right]$$

Substituting the values:

$$Z = \frac{1}{5} \left[ (w - 1.8)^2 + (2w - 3.9)^2 + (3w - 6.1)^2 + (4w - 8.0)^2 + (5w - 10.2)^2 \right]$$

Since $x_1^2 + x_2^2 + x_3^2 + x_4^2 + x_5^2 = 1 + 4 + 9 + 16 + 25 = 55$, this simplifies to:

$$Z = \frac{1}{5} \left( 55w^2 - 2w \sum x_i y_i + \sum y_i^2 \right)$$

Where $\sum x_i y_i = 1(1.8) + 2(3.9) + 3(6.1) + 4(8.0) + 5(10.2) = 110.1$ and $\sum y_i^2 = 1.8^2 + 3.9^2 + 6.1^2 + 8.0^2 + 10.2^2 = 222.1$, giving the **final closed form**:

$$\boxed{Z(w) = \frac{1}{5}(55w^2 - 220.2w + 222.1) = 11w^2 - 44.04w + 44.42}$$

This is the parabola you see in the loss curve — a quadratic in $w$ with its minimum at $w^* = \frac{44.04}{22} \approx 2.0$.
