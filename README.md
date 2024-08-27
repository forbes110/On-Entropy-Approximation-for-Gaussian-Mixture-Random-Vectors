# On-Entropy-Approximation-for-Gaussian-Mixture-Random-Vectors
Unofficial Implementation of paper "On Entropy Approximation for Gaussian Mixture Random Vectors" in python
original paper: https://ieeexplore.ieee.org/document/4648062

### 1. Entropy Approximation for Gaussian Mixture Random Vectors with Taylor expansion

Note that we observe that there are some typos in this paper, so we modified the formulation, the version to be implemented is shown below
```math
$\begin{aligned}
& H(\boldsymbol{x}) \approx-\blue{\sum_{i=1}^L \omega_i }\red{\int_{\mathbb{R}^N}  \mathcal{N}\left({x} ; {\mu}_i, \mathbf{C}_i\right) \cdot\green{
 \left(\sum_{k=0}^R {\frac{1}{k!}\left(\left(\red{x}-\blue{{\mu}_i}\right) \odot \nabla\right)^k \log g({x})}\right) }\mathrm{d} {x}}
\end{aligned}$
```math

```math
The vector-valued scalar function $\begin{aligned}\log g(x) = \log\left(\sum\limits_{i=1}^L \omega_i \cdot \mathcal{N}\left({x}; {\mu}_i, \mathbf{C}_i\right)\right)\end{aligned}$ mapped $\R^d \rightarrow \R$ is approximated by  $\begin{aligned}\sum\limits_{k=0}^R {\frac{1}{k!}\left(\left({x}-{{\mu}_i}\right) \odot \nabla\right)^k \log g({x})}\end{aligned}$, where $R\leq L$ is the truncation order.

The vector-valued scalar function taylor series expansion is by $\begin{aligned}f(x) \approx f(x_0) + \nabla f(x_0) \cdot (x - x_0) + \frac{1}{2} (x - x_0)^\top H(x_0) (x - x_0) + \sum_{k=3}^{n} \frac{1}{k!} D^{(k)}f(x_0) \cdot (x - x_0)^{\otimes k} + \text{h.o.t}\end{aligned}$
```math

### 2. Lower Bound
### 3. Upper Bound

