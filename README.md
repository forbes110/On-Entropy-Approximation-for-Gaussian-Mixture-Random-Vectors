# On-Entropy-Approximation-for-Gaussian-Mixture-Random-Vectors
Unofficial Implementation of paper "On Entropy Approximation for Gaussian Mixture Random Vectors" in python
original paper: https://ieeexplore.ieee.org/document/4648062

### 1. Entropy Approximation for Gaussian Mixture Random Vectors with Taylor expansion

Note that we observe that there are some typos in this paper, so we modified the formulation, the version to be implemented is shown below
```math
\begin{aligned}
& H(\boldsymbol{x}) \approx-{\sum_{i=1}^L \omega_i }{\int_{\mathbb{R}^N}  \mathcal{N}\left({x} ; {\mu}_i, \mathbf{C}_i\right) \cdot{
 \left(\sum_{k=0}^R {\frac{1}{k!}\left(\left({x}-{{\mu}_i}\right) \odot \nabla\right)^k \log\ g({x})}\right) }\mathrm{d} {x}}
\end{aligned}
```

The vector-valued scalar function 
```math 
\begin{aligned} \log\ {g(x)} = \log\ \left(\sum\limits_{i=1}^L \omega_i \cdot \mathcal{N}\left({x}; {\mu}_i, \mathbf{C}_i\right)\right)\end{aligned}
```
mapped $\mathbb{F}^d \rightarrow \mathbb{F}$ is approximated by  

```math
\begin{aligned}\sum\limits_{k=0}^R {\frac{1}{k!}\left(\left({x}-{{\mu}_i}\right) \odot \nabla\right)^k \log\ g({x})}\end{aligned},
```

where $R\leq L$ is the truncation order.

The vector-valued scalar function taylor series expansion is by 

```math
\begin{aligned}f(x) \approx f(x_0) + \nabla f(x_0) \cdot (x - x_0) + \frac{1}{2} (x - x_0)^\top H(x_0) (x - x_0) + \sum_{k=3}^{n} \frac{1}{k!} D^{(k)}f(x_0) \cdot (x - x_0)^{\otimes k} + \text{h.o.t}\end{aligned}
```

### 2. Lower Bound
```math
\begin{aligned}
& H_l({\boldsymbol{x}})=-\sum_{i=1}^L \omega_i \cdot \log \left(\sum_{j=1}^L \omega_j \cdot z_{i, j}\right) \end{aligned}
```
where $z_{i, j}=\mathcal{N}\left({\mu}_i ; {\mu}_j, \mathbf{C}_i+\mathbf{C}_j\right)$
### 3. Upper Bound
```math
\begin{aligned}H_u({\boldsymbol{x}})=\sum_{i=1}^L \omega_i \cdot\left(-\log \omega_i+\frac{1}{2} \log \left((2 \pi e)^N\left|\mathbf{C}_i\right|\right)\right)\end{aligned}
```