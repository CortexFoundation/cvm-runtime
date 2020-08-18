# MRT Generalized Quantization Math Formalization

[TOC]

### Quantization

#### Quantized Buffer Definition

| Symmetry   | Granularity  | Quantized Buffer Definition                                  |
| ---------- | ------------ | ------------------------------------------------------------ |
| Symmetric  | Layer-wise   | $sc_x = \Big(2^{PREC-1}-1\Big) / \max{|Xr|}$                 |
| Symmetric  | Channel-wise | $sc_x[i] = (2^{PREC-1} - 1) / \max{(Xr, \text{axis=ich})}[i]$<br>$\forall i \in [0, shp[ich])$ |
| Zero Point | Layer-wise   | $sc_x = \Big(2^{PREC}-1\Big) / \Big({\text{max}Xr-\text{min}Xr}\Big)$<br>$zp_x = \big\lceil -sc_x \cdot \min{(Xr)} \big\rceil$ |
| Zero Point | Channel-wise | $sc_x[i] = \Big(2^{PREC}-1\Big) / \Big[\max{(Xr, \text{axis=ich})}[i] - \min{(Xr, \text{axis=ich})}[i]\Big]$<br>$zp_x[i] = \big\lceil -sc_x \cdot \min{(Xr, \text{axis=ich})} \big\rceil$<br>$\forall i \in [0, shp[ich])$ |

#### Channel Split

$$
\forall i \in [0, C)
$$

$$
Xei = \text{slice}\Big(Xe, \text{begin=(None,)*i+(i,)+(None,)*(ndims-i-1)}, \text{end=(None,)*i+(i+1,)+(None,)*(ndims-i-1)}\Big)
$$

#### Quantization Process

| Symmetry   | Granularity  | Quantization Process                                         |
| ---------- | ------------ | ------------------------------------------------------------ |
| Symmetric  | Layer-wise   | $Xq = \text{round}\Big( sc_x / sc_{xe} \Big) \cdot Xe$<br>$Wq = \text{round}(sc_w \cdot We)$ |
| Symmetric  | Channel-wise | $Xqi = \text{round}\Big( sc_x[i] / sc_{xe}[i] \Big) \cdot Xei$<br>$Wq = \text{round}(sc_w[i] \cdot Wei)$ |
| Zero Point | Layer-wise   | $Xq = \text{round}\Big(sc_x / sc_{xe}\Big) \cdot Xe + zp_x$<br>$Wq = \text{round}\Big( sc_w \cdot Wr \Big) + zp_w$ |
| Zero Point | Channel-wise | $Xqi = \text{round}\Big( sc_x[i] / sc_{xe}[i] \Big) \cdot Xei + zp_x[i]$<br>$Wqi = \text{round}\Big( sc_w[i] \cdot Wei \Big) + zp_w[i]$ |

### NN Operator Expansion

#### Convolution

For simplicity:

1. Only 2-D case is considered
2. `num_group` is asserted to be 1
3. `bias` as well as `padding` are fused.

Inputs are listed below:

1. Input data $X$, of shape $(N,C,H,W)$

2. Convolution kernel weight $W$, of shape $(O,C,KH,KW)$

Attributes are listed below:

2. $\text{stride} = (SH,SW)$
3. $\text{dilation} = (DH,DW)$

Layer-wise 2-D convolution formalization:
$$
\forall n \in [0, N)
$$

$$
\forall o \in [0, O)
$$

$$
\forall p \in \Bigg{[} 0, \bigg\lfloor \frac{H - DH \cdot (KH-1) - 1}{SH} \bigg\rfloor + 1 \Bigg{)}
$$

$$
\forall q \in \Bigg{[} 0, \bigg\lfloor \frac{W - DW \cdot (KW-1) - 1}{SW} \bigg\rfloor + 1 \Bigg{)}
$$

$$
Yr[n,o,p',q'] = \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xr[n,i,p',q'] \cdot Wr[o,i,ki,kj]
$$

where:
$$
p' = p \cdot SH + ki \cdot DH
$$

$$
q' = q \cdot SW + kj \cdot DW
$$

Expansion scale definition:
$$
\begin{equation}
  sc_{ye} =
    \begin{cases}
      sc_{xe} \cdot sc_{we} & \text{Layer-wise} \\
      \max_{i \in [0, C)}{\big\{sc_{xe}[i] \cdot sc_{we}[i]\big\}} & \text{Channel-wise}
    \end{cases}       
\end{equation}
$$


**Symmetric quantized $X$ and $W$**

Layer-wise:
$$
Ye = \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n,i,p',q'] \cdot Wq[o,i,ki,kj]
$$
Channel-wise:
$$
Ye[n,o,p',q'] = \sum_{i=0}^{C-1} \frac{sc_{ye}}{sc_{xe}[i] \cdot sc_{we}[i]} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqi[n,0,p',q'] \cdot Wqi[o,0,ki,kj]
$$
**Zero point quantized $X$ and $W$**

Layer-wise:
$$
Ye[n,o,p',q']
= \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n,i,p',q'] \cdot Wq[o,i,ki,kj]
- KH \cdot KW \cdot zp_w \sum_{i=0}^{C-1} Xq[n,i,p',q'] \\
- zp_x \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
+ C \cdot KH \cdot KW \cdot zp_x \cdot zp_w
$$
Channel-wise:
$$
Ye[n,o,p',q']
= \sum_{i=0}^{C-1} \frac{sc_{ye}}{sc_{xe}[i] \cdot sc_{we}[i]} \Bigg\{
\sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqi[n,0,p',q'] \cdot Wqi[o,0,ki,kj]
- KH \cdot KW \cdot zp_w[i] \cdot Xqi[n,0,p',q'] \\
- zp_x[i] \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wqi[o,0,ki,kj]
+ KH \cdot KW \cdot zp_x[i] \cdot zp_w[i]
\Bigg\}
$$
**Zero point quantized $X$ and Symmetric quantized $W$**

Layer-wise:
$$
Ye[n,o,p',q'] = 
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n,i,p',q'] \cdot Wq[o,i,ki,kj]
- zp_x \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
$$
Channel-wise:
$$
Ye[n,o,p',q'] = \sum_{i=0}^{C-1} \frac{sc_{ye}}{sc_{xe}[i] \cdot sc_{we}[i]} \Bigg\{ 
\sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqi[n,0,p',q'] \cdot Wqi[o,0,ki,kj]
- zp_x[i] \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wqi[o,0,ki,kj]
 \Bigg\}
$$


**Symmetric quantized $X$ and Zero point quantized $W$**

Layer-wise:
$$
Ye[n,o,p',q'] =  
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n,i,p',q'] \cdot Wq[o,i,ki,kj]
- KH \cdot KW \cdot zp_w \sum_{i=0}^{C-1} Xq[n,i,p',q']
$$
Channel-wise:
$$
Ye[n,o,p',q'] = \sum_{i=0}^{C-1} \frac{sc_{ye}}{sc_{xe}[i] \cdot sc_{we}[i]} \Bigg\{ 
\sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqi[n,0,p',q'] \cdot Wqi[o,0,ki,kj]
- KH \cdot KW \cdot zp_w[i] \sum_{i=0}^{C-1} Xq[n,0,p',q'] 
 \Bigg\}
$$


#### pad

For simplicity:

1. Only 
2. `num_group` is asserted to be 1
3. `bias` as well as `padding` are fused.

#### relu

#### Pooling

#### FullyConnected

### Broadcast Operator Expansion

#### broadcast_add