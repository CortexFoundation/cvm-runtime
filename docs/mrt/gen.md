# MRT Generalized Quantization Math Formalization

[TOC]

### Quantizer

#### Uniform Symmetric Quantizer

$$
sc_{x} = \frac{2^{PREC-1}-1}{\max{|Xr|}}
\tag{Scale}
$$

$$
\begin{align}
frac, exp = \text{cvm_float}\bigg(\frac{sc_{x}}{sc_{xe}}\bigg)\\
Xq = \text{realize} (X_{e}, frac, exp)
\end{align}
\tag{Input Quantization}
$$

$$
Wq = \text{round}\Big(sc_{w} \cdot Wr\Big)
\tag{Weight Quantization}
$$

#### Uniform Affine Quantizer

$$
sc_{x} = \frac{2^{PREC}-1}{\text{max}Xr-\text{min}Xr}
\tag{Scale}
$$

$$
zp_{xe} = \text{round} \Big(\min Xr \cdot sc_{xe}\Big)
\tag{Zero Point of Input}
$$

$$
\begin{align}
frac, exp = \text{cvm_float}\bigg(\frac{sc_{x}}{sc_{xe}}\bigg) \\
Xq = \text{realize}(Xe - zp_{xe}, frac, exp)
\end{align}
\tag{Input Quantization}
$$

$$
zp_{wr} = \min Wr
\tag{Zero Point of Weight}
$$

$$
W_{q} = \text{round} \Big[ (sc_{w} \left( W_{r} - zp_{wr} \right) \Big]
\tag{Weight Quantization}
$$

### Re-quantization

$$
Xr = \frac{Xq}{sc_{x}}
\tag{Symmetric Re-quantization}
$$

$$
Xr = \frac{Xq + zp_{x}}{sc_{x}}
\tag{Zero-point Re-quantization}
$$

Re-quantization is elaborated in operator expansion, see [NN Operator Expansion](#nn-operator-expansion), [Broadcast Operator Expansion](#broadcast-operator-expansiono), [Elemwise Operator Expansion](#elemwise-operator-expansion), [Transform Operator Expansion](#transform-operator-expansion).

### Granularity

MRT Support both layer-wise and channel-wise quantization. Channel wise quantization is implemented by graph-level channel split and channel merge. 

#### Channel Split

To compromise between precision and calculation operations, MRT support quantization with respect to channel features. Use `slice` to split the channels in MRT rewrite process:
$$
\forall i \text{ in } [0, C, \text{step})
$$

$$
Xi = \text{slice}\Big(X, \\
\text{begin=(None,)*ichannel+(i,)+(None,)*(ndims-ichannel-1)}, \\
\text{end=(None,)*ichannel+(i+step,)+(None,)*(ndims-ichannel-1)}, \\
\text{step=(-1,)*ichannel+(step,)+(-1,)*(ndims-ichannel-1)}\Big)
$$

If $X$ is of channel feature and $W$ is of layer feature or vice versa, $W$ (or $X$) will also be split to be compatible with $X$ (or $W$).

Take `Convolution` for instance (only uniform symmetric quantization is considered for simplicity), layer-wise Convolution can be rewritten as:
$$
\begin{align}

Ye[n,o,p,q] 

&= \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj] \\

&= \sum_{i=0}^{C/step-1} \sum_{j=i*step}^{(i+1)*step-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, j, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,j,ki,kj] \\

&= \sum_{i=0}^{C/step-1} Convolution(Xq[:,i*step:(i+1)*step,:,:], Wq[:,i*step:(i+1)*step,:,:]) \\

&= \sum_{i=0}^{C/step-1} Yei[n,o,p,q]

\end{align}
$$

Note, if `num_groups` is not 1, then slice channel is not possible.

Specifically, suppose kernel weight $W$ is of shape $(O,IC,KH,KW)$ and input data $X$ is of shape $(N,C,H,W)$.
$$
Yr[n,o,p,q]

= \sum_{i=0}^{IC-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xr\Bigg[n, \bigg\lfloor \frac{o \cdot IC}{OPG} \bigg\rfloor + i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW\Bigg] \cdot Wr[o,i,ki,kj]
$$

#### Channel Merge

Merge the channel symbol components to the equivalent symbol. 

For operators like `Convolution`, merge as follows:
$$
X = \sum_{i=0}^{C-1} Xi
\tag{add_n}
$$
For operators like `pad`, `relu`, `Pooling`, merge as follows:
$$
X = \text{concat}\Big( \big[Xi, \forall i \in [0, C) \big], \text{dim=ich} \Big)
\tag{concat}
$$

### NN Operator Expansion

#### Convolution

**Limitations**

1. Only 2-D case is considered
2. `num_group` is asserted to be 1
3. `bias` is fused in MRT rewrite

**Inputs**

1. Input data $X$, of shape $(N,C,H,W)$

2. Kernel weight $W$, of shape $(O,C,KH,KW)$

**Attributes**

1. $\text{padding} = (PH,PW)$

2. $\text{stride} = (SH,SW)$
3. $\text{dilation} = (DH,DW)$

**Real Formalization**
$$
\forall n \in [0, N), \quad
\forall o \in [0, O), \quad
\forall p \in \Bigg{[} 0, \bigg\lfloor \frac{H - DH \cdot (KH-1) - 1}{SH} \bigg\rfloor + 1 \Bigg{)}, \quad
\forall q \in \Bigg{[} 0, \bigg\lfloor \frac{W - DW \cdot (KW-1) - 1}{SW} \bigg\rfloor + 1 \Bigg{)}
$$

$$
Yr[n,o,p,q] = \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xr[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wr[o,i,ki,kj]
$$

**Expansion Scale**
$$
sc_{ye} = sc_{x} \cdot sc_{w}
$$
**Padding beforehand**

```python
Xe = pad(Xe, mode="constant", pad_width=(0,0,0,0,PH,PH,PW,PW), constant_value=0)
```

**Expansion Formalization 1: Symmetric Quantized X and W**
$$
Ye[n,o,p,q] = \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj]
$$

```python
Ye = Convoltion(Xq, Wq, **attrs)
infer_prec = get_bit_cnt(C*KH*KW) + xprec + wprec
```

**Expansion Formalization 2: Zero Point Quantized X and Symmetric Quantized W**
$$
\begin{equation} \begin{split}
Ye[n,o,p,q] = 
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj] \\
+ x_{zp} \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
\end{split} \end{equation}
$$
```python
Ye = Convoltion(Xq, Wq, **attrs) + C
infer_prec1 = get_bit_cnt(C*KH*KW) + xprec + wprec + 1
infer_prec2 = get_bit_cnt(abs(xzp)*C*KH*KW) + wprec
infer_prec = max(infer_prec1, infer_prec2) + 1
```

**Expansion Formalization 3: Symmetric Quantized X and Zero Point Quantized W**
$$
\begin{equation} \begin{split}
Ye[n,o,p,q] =  
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj] \\
+ w_{zp} \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW]
\end{split} \end{equation}
$$
```python
Ye = Convoltion(Xq, Wq, **attrs) + C * Convolution(Xq, W1, **attrs)
infer_prec1 = get_bit_cnt(C*KH*KW) + xprec + wprec + 1
infer_prec2 = get_bit_cnt(abs(wzp)*C*KH*KW) + xprec
infer_prec = max(infer_prec1, infer_prec2) + 1
```

**Expansion Formalization 4: Zero Point Quantized X and W**
$$
\begin{equation} \begin{split}
Ye[n,o,p,q]
= \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj] \\
+ w_{zp} \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \\
+ x_{zp} \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
+ C \cdot KH \cdot KW \cdot x_{zp} \cdot w_{zp}
\end{split} \end{equation}
$$

```python
Ye = Convoltion(Xq, Wq, **attrs) + wzp * Convoltion(Xq, W1, **attrs) + C2 + C3
infer_prec1 = get_bit_cnt(C*KH*KW) + xprec + wprec + 2
infer_prec2 = get_bit_cnt(abs(wzp)*C*KH*KW) + xprec + 1
infer_prec3 = get_bit_cnt(abs(xzp)*C*KH*KW) + wprec + 1
infer_prec4 = get_bit_cnt(abs(wzp)*abs(xzp)*C*KH*KW)
infer_prec = max(infer_prec1, infer_prec2, infer_prec3, infer_prec4) + 2
```

#### pad

**Limitations**

1. Only support `constant` mode
2. $\text{constant_value} = 0$
3. Only support pad of $H$ dimension and $W$ dimension

**Inputs**

1. Input data $X$, of shape $(N,C,H,W)$

**Attributes**

1. $\text{pad_width} = (0,0,0,0,PH_1,PH_2,PW_1,PW_2)$

**Real Formalization**
$$
\forall n \in [0, N), \quad
\forall c \in [0, C), \quad
\forall h \in [0, PH_1 + H + PH_2), \quad
\forall w \in [0, PW_1 + W + PW_2)
$$


$$
\begin{equation}
  Yr[n,c,h,w] =
    \begin{cases}
      Xr[n, c, h-PH_1, w-PW_1] & PH_1 \leq h<H+PH_1 \wedge PW_1 \leq w < W+PW_1 \\
      0 & \text{otherwise}
    \end{cases}       
\end{equation}
$$
**Expansion Scale**
$$
sc_{ye} = sc_{xe}
$$
**Expansion Formalization**
$$
\begin{equation} \begin{split}
  Ye[n,c,h,w] =
    \begin{cases}
      Xe[n, c, h-PH_1, w-PW_1] & PH_1 \leq h<H+PH_1 \wedge PW_1 \leq w < W+PW_1 \\
      0 & \text{otherwise}
    \end{cases}       
\end{split} \end{equation}
$$

```python
Ye = pad(Xe, **attrs)
```

#### relu

**Inputs**

1. Input data $X$, of shape $(M_0,M_1,...,M_{N-1})$

**Real Formalization**
$$
\forall i \in [0, N), \quad \forall m_i \in [0, M_i)
$$

$$
Yr[m_0,m_1,...,m_{N-1}] = \max{\big( 0, Xr[m_0,m_1,...,m_{N-1}] \big)}
$$

**Expansion Scale**
$$
sc_{ye} = sc_{xe}
$$
**Expansion Formalization**
$$
Ye[m_0,m_1,...,m_{N-1}] = \max{\big(0, Xe[m_0,m_1,...,m_{N-1}]\big)}
$$

```python
Ye = relu(Xe)
```

#### Pooling

 **Limitations**

1. Only 2-D case is considered
2.  `avg` pooling will be rewritten into `Convolution` or `broadcast_mul`
3. Only `max` pooling will be considered

**Inputs**

1. Input data $X$, of shape $(N,C,H,W)$


**Attributes**

1. $\text{stride} = (SH,SW)$

2. $\text{kernel} = (KH, KW)$
3. $\text{padding} = (PH, PW)$

**Real Formalization**
$$
\forall n \in [0, N), \quad
\forall c \in [0, C), \quad
\forall p \in \Bigg{[} 0, \bigg\lfloor \frac{H - KH}{SH} \bigg\rfloor + 1 \Bigg{)}, \quad
\forall q \in \Bigg{[} 0, \bigg\lfloor \frac{W - KW}{SW} \bigg\rfloor + 1 \Bigg{)}
$$

$$
Yr[n,c,p,q] = \max_{p \cdot SH \leq p' < p \cdot SH + KH \\
q \cdot SW \leq q' < q \cdot SW + KW} 
Xr[n,c,p',q']
$$

**Padding beforehand**

```python
Xe = pad(Xe, mode="constant", pad_width=(0,0,0,0,PH,PH,PW,PW), constant_value=INT_MIN)
```

**Expansion Scale**
$$
sc_{ye} = sc_{xe}
$$
**Expansion Formalization**
$$
Ye[n,c,p,q] = \max_{p' \in [p \cdot SH, p \cdot SH + KH) \\
q' \in [q \cdot SW, q \cdot SW + KW)} 
Xe[n,c,p',q']
$$

```python
Ye = Pooling(Xe, stride=stride, kernel=kernel)
```

#### FullyConnected

**Limitations**

1. The input only supports layer-wise quantization
2. `bias` is fused in MRT rewrite

**Input**

1. Input data $X$, of shape $(N,K)$
2. Weight $W$, of shape $(M, K)$

**Real Formalization**
$$
\forall m \in [0, M), \quad
\forall n \in [0, N)
$$

$$
Yr[n,m] = \sum_{i=0}^{K-1} X[n,i] \cdot W[m,i]
$$

**Expansion Scale**
$$
sc_{ye} = sc_{x} \cdot sc_{w}
$$
**Expansion Formalization 1: Symmetric Quantized X and W**
$$
Ye[n,m] = \sum_{i=0}^{K-1} Xq[n,i] \cdot Wq[m,i]
$$

```python
Ye = FullyConnected(Xq, Wq)
infer_prec = get_bit_cnt(K) + xprec + wprec
```

**Expansion Formalization 2: Zero Point Quantized X​ and Symmetric Quantized W**
$$
Ye[n,m] = \sum_{i=0}^{K-1} Xq[n,i] \cdot Wq[m,i] 
+ zp_x \sum_{i=0}^{K-1} Wq[m,i]
$$

```python
Ye = FullyConnected(Xq, Wq) + C
infer_prec1 = get_bit_cnt(K) + xprec + wprec + 1
infer_prec2 = get_bit_cnt(abs(C))
infer_prec = max(infer_prec1, infer_prec2) + 1
```

**Expansion Formalization 3: Symmetric Quantized X​ and Zero Point Quantized W**
$$
Ye[n,m] = \sum_{i=0}^{K-1} Xq[n,i] \cdot Wq[m,i] +
zp_{w} \sum_{i=0}^{K-1} Xq[n,i]
$$

```python
Ye = FullyConnected(Xq, Wq) + C * sum(Xq, axis=1, keep_dims=True)
infer_prec1 = get_bit_cnt(K) + xprec + wprec + 1
infer_prec2 = get_bit_cnt(abs(C)*K) + xprec
infer_prec = max(infer_prec1, infer_prec2) + 1
```

**Expansion Formalization 4: Zero Point Quantized X​ and W**
$$
Ye[n,m] = \sum_{i=0}^{K-1} Xq[n,i] \cdot Wq[m,i]
+ zp_{w} \sum_{i=0}^{K-1} Xq[n,i]
+ zp_{x} \sum_{i=0}^{K-1} Wq[m,i]
+ zp_{w} \cdot zp_{x} \cdot K
$$

```python
Ye = FullyConnected(Xq, Wq) + C1 * sum(Xq, axis=1, keep_dims=True) + C2 + C3
infer_prec1 = get_bit_cnt(K) + xprec + wprec + 2
infer_prec2 = get_bit_cnt(abs(C1)*K) + xprec + 1
infer_prec3 = get_bit_cnt(abs(max(C2)))
infer_prec4 = get_bit_cnt(abs(C3))
infer_prec = max(infer_prec1, infer_prec2, infer_prec3, infer_prec4) + 2
```

### Broadcast Operator Expansion

#### broadcast_add

use [Quantize Scale](#quantize-scale).

### Elemwise Operator Expansion

#### elemwise_add

use [Quantize Scale](#quantize-scale).

#### add_n

use [Quantize Scale](#quantize-scale).

### Transform Operator Expansion

#### concat

use [Quantize Scale](#quantize-scale).

#### flatten

**Inputs**

1. Input data $X$, of shape $(M_0,M_1,...,M_{N-1})$

**Real Formalization**
$$
\forall i \in [0, N), \quad \forall m_i \in [0, M_i)
$$

$$
Yr\Bigg[\sum_{j=0}^{N-2} m_j \prod_{i=j+1}^{N-1}M_i + m_{N-1}\Bigg] = Xr[m_0,m_1,...,m_{N-1}]
$$

**Expansion Scale**
$$
sc_{ye} = sc_{xe}
$$
**Expansion Formalization**
$$
Ye\Bigg[\sum_{j=0}^{N-2} m_j \prod_{i=j+1}^{N-1}M_i + m_{N-1}\Bigg] = Xe[m_0,m_1,...,m_{N-1}]
$$

```python
Ye = flatten(Xe)
```

### Generalized Expansion Function

#### Quantize Scale

**Limitations**

1. All the inputs only support symmetric quantize

**Expansion Scale**
$$
sc_{ye} = sc_{xi}
$$
**Expansion Formalization**

```python
Ye = merge(**Xqs, **attrs)
infer_prec = max(xprecs) if op_name == "Concat" else max(xprecs)+1
```
