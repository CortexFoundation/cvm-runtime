# MRT Generalized Quantization Math Formalization



### Table of Contents
[Quantize and Re-quantize Process](#Quantize and Re-quantize Process)
  [Symmetric Layer-wise](#Symmetric Layer-wise)
  [Zero Point Layer-wise](#Zero Point Layer-wise)
[Multiplication Operations](#Multiplication Operations)
  [2D Convolution](#2D Convolution)
[Addition Operations](#Addition Operations)


### Quantize and Re-quantize Process

#### Symmetric Layer-wise

Scale Definition:
$$
sc = \frac{2^{PREC-1} - 1}{\max{(|x|)}}
$$


Quantize formalization:
$$
Xq = \left\lfloor sc \cdot X \right\rfloor
$$
Re-quantize formalization:
$$
X = \frac{Xq}{sc}
$$

#### Symmetric Channel-wise

Scale definition:
$$
sc = \frac{2^{PREC}-1}{\text{max}(x)-\text{min}(x)}
$$
Zero point definition:
$$
zp = \frac{\text{min}(x)\left( 2^{PREC} - 1\right)}{\text{max}(x) - \text{min}(x)}
$$
Quantize formalization:
$$
Xq = \left\lfloor sc \cdot X-zp \right\rfloor
$$
Re-quantization:
$$
X = \frac{Xq + zp}{sc}
$$

#### Zero Point Layer-wise



#### Zero Point Channel-wise




### Multiplication Operations

#### 2D Convolution

Inputs are listed below:

1. Input data $X$, of shape $(N,C,H,W)$

2. Convolutional kernel weight $W$, of shape $(OC,IC,KH,KW)$

Attributes are listed below:

1. $padding = (PH, PW)$
2. $stride = (SH,SW)$
3. $dilation = (DH,DW)$
4. $groups$,  $C = IC \cdot groups$, $OC = OPG \cdot groups$

(For simplicity, bias is not discussed in the formulization of `Convolution`.)

2D convolution formalization:
$$
\forall n \in [0, N)
$$

$$
\forall oc \in [0, OC)
$$

$$
\forall p \in \Bigg{[} 0, \bigg\lfloor \frac{H + 2\cdot PH - DH \cdot (KH-1) - 1}{SH} \bigg\rfloor + 1 \Bigg{)}
$$

$$
\forall q \in \Bigg{[} 0, \bigg\lfloor \frac{W + 2\cdot PW - DW \cdot (KW-1) - 1}{SW} \bigg\rfloor + 1 \Bigg{)}
$$

$$
Y[n,oc,p,q] = \sum_{ic=0}^{IC-1} \text{kernel}\left(n,\lfloor oc/OPG \rfloor \cdot IC + ic,p,q,oc,ic\right)
$$

2D kernel function formalization:
$$
\text{kernel}(n,j,p,q,o,i) = \sum_{ki=0}^{KH} \sum_{kj=0}^{KW} \text{pad}\big(n, j, p \cdot SH - PH + ki \cdot DH, q \cdot SW - PW + kj \cdot DW\big) \cdot W[o,i,ki,kj]
$$

2D pad function formalization:
$$
\begin{equation}
  \text{pad}(n,j,p,q) =
    \begin{cases}
      X[n,j,p,q] & \forall p \in [0,H), \forall q \in [0,W) \\
      0 & \text{otherwise}
    \end{cases}       
\end{equation}
$$

1. Symmetric Layer-wise quantized $X$ and $W$

$$

$$



### Addition Operations

