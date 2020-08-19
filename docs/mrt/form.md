# MRT Generalized Quantization Math Formalization

[TOC]

### Quantization

#### Quantized Buffer Definition

| Symmetry   | Granularity  | Quantized Buffer Definition                                  |
| ---------- | ------------ | ------------------------------------------------------------ |
| Symmetric  | Layer-wise   | $sc_x = \Big(2^{PREC-1}-1\Big) / \max{|Xr|}$                 |
| Symmetric  | Channel-wise | $sc_{xi} = (2^{PREC-1} - 1) / \max{(|Xr|, \text{axis=ich})}[i]$<br>$\forall i \in [0, shp[ich])$ |
| Zero Point | Layer-wise   | $sc_x = \Big(2^{PREC}-1\Big) / \Big({\text{max}Xr-\text{min}Xr}\Big)$<br>$zp_x = \big\lceil -sc_x \cdot \min{(Xr)} \big\rceil$ |
| Zero Point | Channel-wise | $sc_{xi} = \Big(2^{PREC}-1\Big) / \Big[\max{(Xr, \text{axis=ich})}[i] - \min{(Xr, \text{axis=ich})}[i]\Big]$<br>$zp_{xi} = \big\lceil -sc_{xi} \cdot \min{(Xr, \text{axis=ich})} \big\rceil$<br>$\forall i \in [0, shp[ich])$ |

#### Channel Lazy Split

To compromise between precision and calculation operations, MRT support quantization with respect to channel features. Use `slice` to split the channels:
$$
\forall i \in [0, C)
$$

$$
Xei = \text{slice}\Big(Xe, \text{begin=(None,)*i+(i,)+(None,)*(ndims-i-1)}, \text{end=(None,)*i+(i+1,)+(None,)*(ndims-i-1)}\Big)
$$

The split channels all have the same scale:
$$
sc_{xei} = sc_{xe}
$$

#### Channel Lazy Merge

Merge the channel symbol components for the purpose of latter-layer quantization. Adopt **lazy merge** to reduce precision lost. The scale of the merged merged operator can be calculated as:
$$
sc_{xe} = \max_{\forall i \in [0,C)}{sc_{xei}}
$$
For operators like `Convolution`, use `broadcast_add` to merge:
$$
Xe = \sum_{i=0}^{C-1} \frac{sc_{xe}}{sc_{xei}} Xei
\tag{broadcast_add}
$$
For operators like `pad`, use `concat` to merge:
$$
Xe = \text{concat}\Bigg( \Bigg[\frac{sc_{xe}}{sc_{xei}}Xei,i=0,1,...,C-1\Bigg],
\text{dim=ich}
\Bigg)
\tag{concat}
$$

#### Quantization Process

| Symmetry   | Granularity  | Quantization Process                                         |
| ---------- | ------------ | ------------------------------------------------------------ |
| Symmetric  | Layer-wise   | $Xq = \text{round}\Big( sc_x / sc_{xe} \Big) \cdot Xe$<br>$Wq = \text{round}(sc_w \cdot We)$ |
| Symmetric  | Channel-wise | $Xqi = \text{round}\Big( sc_{xi} / sc_{xei} \Big) \cdot Xei$<br>$Wq = \text{round}(sc_{wi} \cdot Wei)$ |
| Zero Point | Layer-wise   | $Xq = \text{round}\Big(sc_x / sc_{xe}\Big) \cdot Xe + zp_x$<br>$Wq = \text{round}\Big( sc_w \cdot Wr \Big) + zp_w$ |
| Zero Point | Channel-wise | $Xqi = \text{round}\Big( sc_{xi} / sc_{xei} \Big) \cdot Xei + zp_{xi}$<br>$Wqi = \text{round}\Big( sc_{wi} \cdot Wei \Big) + zp_{wi}$ |

### Expansion Scale

Expansion scale definition for operators like `Convolution`:
$$
sc_{ye} = sc_{x} \cdot sc_{w}
\tag{Layer-wise}
$$

$$
sc_{yei} = sc_{xi} \cdot sc_{wi}
\tag{Channel-wise}
$$

Expansion scale definition for operators like `pad`, `relu`, `Pooling`:
$$
sc_{ye} = sc_{x}
\tag{Layer-wise}
$$

$$
sc_{yei} = sc_{xi}
\tag{Channel-wise}
$$

### Expansion Infer Precision

### NN Operator Expansion Formalization

#### Convolution

**Limitation**

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

**Formalization**
$$
\forall n \in [0, N), \quad
\forall o \in [0, O), \quad
\forall p \in \Bigg{[} 0, \bigg\lfloor \frac{H - DH \cdot (KH-1) - 1}{SH} \bigg\rfloor + 1 \Bigg{)}, \quad
\forall q \in \Bigg{[} 0, \bigg\lfloor \frac{W - DW \cdot (KW-1) - 1}{SW} \bigg\rfloor + 1 \Bigg{)}
$$

$$
Yr[n,o,p,q] = \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xr[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wr[o,i,ki,kj]
$$

**Padding beforehand**

```python
# Layer-wise
Xe = pad(Xe, mode="constant", pad_width=(0,0,0,0,PH,PH,PW,PW), constant_value=0)
# Channel-wise
Xei = pad(Xei, mode="constant", pad_width=(0,0,0,0,PH,PH,PW,PW), constant_value=0)
```

**Symmetric quantized $X$ and $W$**
$$
Ye[n,o,p,q] = \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj]
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = Convoltion(Xq, Wq, stride=(SH,SW), dilation=(DH,DW))
```

$$
Yei[n,o,p,q] = \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqi[n, 0, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wqi[o,0,ki,kj]
\tag{Channel-wise}
$$

```python
# Channel-wise
Yei = Convoltion(Xqi, Wqi, stride=(SH,SW), dilation=(DH,DW))
```

**Zero point quantized $X$ and Symmetric quantized $W$**
$$
\begin{equation} \begin{split}
Ye[n,o,p,q] = 
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj] \\
+ zp_x \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
\end{split} \end{equation}
\tag{Layer-wise}
$$
```python
# Layer-wise
Ye = Convoltion(Xq, Wq, stride=(SH,SW), dilation=(DH,DW)) + C
```

$$
\begin{equation} \begin{split}
Yei[n,o,p,q] = \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqi[n, 0, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wqi[o,0,ki,kj] \\
+ zp_{xi} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wqi[o,0,ki,kj]
\end{split} \end{equation}
\tag{Channel-wise}
$$

```python
# Channel-wise
Yei = Convoltion(Xqi, Wqi, stride=(SH,SW), dilation=(DH,DW)) + C
```

**Symmetric quantized $X$ and Zero point quantized $W$**
$$
\begin{equation} \begin{split}
Ye[n,o,p,q] =  
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj] \\
+ zp_w \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW]
\end{split} \end{equation}
\tag{Layer-wise}
$$
```python
# Layer-wise
Ye = Convoltion(Xq, Wq, stride=(SH,SW), dilation=(DH,DW)) + \
	C * Convolution(Xq, W1, stride=(SH,SW), dilation=(DH,DW))
```

$$
\begin{equation} \begin{split}
Yei[n,o,p,q] = 
\sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqi[n, 0, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wqi[o,0,ki,kj] \\
+ zp_{wi} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, 0, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW]
\end{split} \end{equation}
\tag{Channel-wise}
$$

```python
# Channel-wise
Yei = Convoltion(Xqi, Wqi, stride=(SH,SW), dilation=(DH,DW)) + \
	C * Convolution(Xqi, W1, stride=(SH,SW), dilation=(DH,DW))
```

**Zero point quantized $X$ and $W$**
$$
\begin{equation} \begin{split}
Ye[n,o,p,q]
= \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj] \\
+ zp_w \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \\
+ zp_x \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
+ C \cdot KH \cdot KW \cdot zp_x \cdot zp_w
\end{split} \end{equation}
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = Convoltion(Xq, Wq, stride=(SH,SW), dilation=(DH,DW)) + \
	C1 * Convoltion(Xq, W1, stride=(SH,SW), dilation=(DH,DW)) + C2
```

$$
\begin{equation} 
\begin{split}
Yei[n,o,p,q] =
\sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqi[n, 0, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wqi[o,0,ki,kj] \\
+ zp_{wi} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqi[n, 0, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \\
+ zp_{xi} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wqi[o,0,ki,kj]
+ KH \cdot KW \cdot zp_{xi} \cdot zp_{wi}
\end{split}
\end{equation} 
\tag{Channel-wise}
$$

```python
# Channel-wise
Yei = Convoltion(Xqi, Wqi, stride=(SH,SW), dilation=(DH,DW)) + \
	C1 * Convolution(Xqi, W1, stride=(SH,SW), dilation=(DH,DW)) + C2
```

#### pad

**Limitation**

1. Only support `constant` mode
2. $\text{constant_value} = PV$
3. Only support pad of $H$ dimension and $W$ dimension

**Inputs**

1. Input data $X$, of shape $(N,C,H,W)$


**Attributes**

1. $\text{pad_width} = (0,0,0,0,PH_1,PH_2,PW_1,PW_2)$

**Formalization**
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
      PV & \text{otherwise}
    \end{cases}       
\end{equation}
$$

**Symmetric quantized $X$**
$$
\begin{equation} \begin{split}
  Ye[n,c,h,w] =
    \begin{cases}
      Xq[n, c, h-PH_1, w-PW_1] & PH_1 \leq h<H+PH_1 \wedge PW_1 \leq w < W+PW_1 \\
      \text{round}\big(PV \cdot sc_{x}\big) & \text{otherwise}
    \end{cases}       
\end{split} \end{equation}
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = pad(Xq, mode="constant", pad_width=(0,0,0,0,PH1,PH2,PW1,PW2), constant_value=round(PV*sc_x))
```

$$
\begin{equation} \begin{split}
  Yei[n,0,h,w] =
    \begin{cases}
      Xqi[n, 0, h-PH_1, w-PW_1] & PH_1 \leq h<H+PH_1 \wedge PW_1 \leq w < W+PW_1 \\
      \text{round}\big(PV \cdot sc_{xi}\big) & \text{otherwise}
    \end{cases}       
\end{split} \end{equation}
\tag{Channel-wise}
$$

```python
# Channel-wise
Yei = pad(Xqi, mode="constant", pad_width=(0,0,0,0,PH1,PH2,PW1,PW2), constant_value=round(PV*sc_xi))
```

**Zero point quantized $X$**
$$
\begin{equation} \begin{split}
  Ye[n,c,h,w] =
    \begin{cases}
      Xq[n, c, h-PH_1, w-PW_1] + zp_{x} & PH_1 \leq h<H+PH_1 \wedge PW_1 \leq w < W+PW_1 \\
      \text{round}\big(PV \cdot sc_{x}\big) & \text{otherwise}
    \end{cases}       
\end{split} \end{equation}
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = pad(Xq+C, mode="constant", pad_width=(0,0,0,0,PH1,PH2,PW1,PW2), constant_value=round(PV*sc_x))
```

$$
\begin{equation} \begin{split}
  Yei[n,0,h,w] =
    \begin{cases}
      Xqi[n, 0, h-PH_1, w-PW_1] + zp_{xi} & PH_1 \leq h<H+PH_1 \wedge PW_1 \leq w < W+PW_1 \\
      \text{round}\big(PV \cdot sc_{xi}\big) & \text{otherwise}
    \end{cases}       
\end{split} \end{equation}
\tag{Channel-wise}
$$

```python
# Channel-wise
Yei = pad(Xqi+C, mode="constant", pad_width=(0,0,0,0,PH1,PH2,PW1,PW2), constant_value=round(PV*sc_xi))
```

#### relu

**Inputs**

1. Input data $X$, of shape $(M_0,M_1,...,M_{N-1})$

**Formalization**
$$
\forall i \in [0, N), \quad \forall m_i \in [0, M_i)
$$

$$
Yr[m_0,m_1,...,m_{N-1}] = \max{\big( 0, Xr[m_0,m_1,...,m_{N-1}] \big)}
$$

**Symmetric quantized $X$**
$$
Ye[m_0,m_1,...,m_{N-1}] = \max{\big(0, Xq[m_0,m_1,...,m_{N-1}]\big)}
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = relu(xq)
```

$$
Yei[m_0,m_1,...,m_{N-1}] = \max{\big(0, Xqi[m_0,m_1,...,m_{N-1}] +zp_{xi}\big)}
\tag{Channel-wise}
$$

```python
# Channel-wise
Yei = relu(xqi)
```

**Zero point quantized $X$**
$$
Ye[m_0,m_1,...,m_{N-1}] = \max{\big(0, Xq[m_0,m_1,...,m_{N-1}] +zp_{x}\big)}
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = relu(xq+C)
```

$$
Yei[m_0,m_1,...,0,...,m_{N-1}] = \max{\big(0, Xqi[m_0,m_1,...,0,...,m_{N-1}] +zp_{xi}\big)}
\tag{Channel-wise}
$$

```python
# Channel-wise
Yei = relu(Xqi+C)
```

#### Pooling

 **Limitation**

1. Only 2-D case is considered
2.  `avg` pooling will be rewritten into `Convolution` or `broadcast_mul`
3. Only `max` pooling will be considered.

**Inputs**

1. Input data $X$, of shape $(N,C,H,W)$


**Attributes**

1. $\text{stride} = (SH,SW)$

2. $\text{kernel} = (KH, KW)$

**Formalization**
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
# Layer-wise
Xe = pad(Xe, mode="constant", pad_width=(0,0,0,0,PH,PH,PW,PW), constant_value=INT_MIN)
# Channel-wise
Xei = pad(Xei, mode="constant", pad_width=(0,0,0,0,PH,PH,PW,PW), constant_value=INT_MIN)
```

**Symmetric quantized $X$**
$$
Ye[n,c,p,q] = \max_{p' \in [p \cdot SH, p \cdot SH + KH) \\
q' \in [q \cdot SW, q \cdot SW + KW)} 
Xq[n,c,p',q']
$$

```python
# Layer-wise
Ye = Pooling(Xq, stride=(SH,SW), kernel=(KH,KW))
```

$$
Yei[n,0,p,q] = \max_{p' \in [p \cdot SH, p \cdot SH + KH) \\
q' \in [q \cdot SW, q \cdot SW + KW)} 
Xqi[n,0,p',q']
$$

```python
# Channel-wise
Yei = Pooling(Xqi, stride=(SH,SW), kernel=(KH,KW))
```

**Zero point quantized $X$**
$$
Ye[n,c,p,q] = zp_x +
\max_{p' \in [p \cdot SH, p \cdot SH + KH) \\
q' \in [q \cdot SW, q \cdot SW + KW)} 
Xq[n,c,p',q']
$$

```python
# Layer-wise
Ye = C + Pooling(Xq, stride=(SH,SW), kernel=(KH,KW))
```

$$
Yei[n,0,p,q] = zp_{xi} +
\max_{p' \in [p \cdot SH, p \cdot SH + KH) \\
q' \in [q \cdot SW, q \cdot SW + KW)} 
Xqi[n,0,p',q']
$$

```python
# Channel-wise
Yei = C + Pooling(Xqi, stride=(SH,SW), kernel=(KH,KW))
```

#### FullyConnected

**Input**

1. Input data $X$, of shape $(N,K)$
2. Weight $W$, of shape $(M, K)$

**Formalization**
$$
\forall m \in [0, M), \quad
\forall n \in [0, N)
$$

$$
Yr[n,m] = \sum_{i=0}^{K-1} X[n,i] \cdot W[m,i]
$$

**Symmetric quantized $X$ and $W$**
$$
Ye[n,m] = \sum_{i=0}^{K-1} Xq[n,i] \cdot Wq[m,i]
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = FullyConnected(Xq, Wq)
```

**Zero point quantized $X$ and Symmetric quantized $W$**
$$
Ye[n,m] = \sum_{i=0}^{K-1} Xq[n,i] \cdot Wq[m,i] 
+ zp_x \sum_{i=0}^{K-1} Wq[m,i]
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = FullyConnected(Xq, Wq) + C
```

**Symmetric quantized $X$ and Zero point quantized $W$**
$$
Ye[n,m] = \sum_{i=0}^{K-1} Xq[n,i] \cdot Wq[m,i] +
zp_{w} \sum_{i=0}^{K-1} Xq[n,i]
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = FullyConnected(Xq, Wq) + C * sum(Xq, axis=1, keep_dims=True)
```

**Zero point quantized $X$ and $W$**
$$
Ye[n,m] = \sum_{i=0}^{K-1} Xq[n,i] \cdot Wq[m,i]
+ zp_{w} \sum_{i=0}^{K-1} Xq[n,i]
+ zp_{x} \sum_{i=0}^{K-1} Wq[m,i]
+ zp_{w} \cdot zp_{x} \cdot K
\tag{Layer-wise}
$$

```python
# Layer-wise
Ye = FullyConnected(Xq, Wq) + C1 * sum(Xq, axis=1, keep_dims=True) + C2
```

### Broadcast Operator Expansion

#### broadcast_add

#### broadcast_mul

### Transform Operator Expansion

#### concat

#### flatten