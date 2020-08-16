# MRT Generalized Quantization Math Formalization

[TOC]

### Quantize and Re-quantize Process

#### Symmetric Layer-wise

Scale Definition:
$$
sc_x = \frac{2^{PREC-1} - 1}{\max{(|x|)}}
$$


Float-to-int quantize formalization:
$$
Xq = \left\lfloor sc_x \cdot X \right\rfloor
$$
Int-to-float re-quantize formalization:
$$
Xr = \frac{Xq}{sc_x}
$$

Int-propagation quantize formalization:
$$
Xqn = \Bigg\lfloor \frac{sc_{xn}}{sc_x} Xq \Bigg\rfloor
$$

$$
Xqn \approx \text{cvm_shift}(frac \cdot Xq, sb)
$$

where:
$$
frac, sb = \text{cvm_float}\left(rsc_{xn}\right)
$$

$$
rsc_{xn} = \frac{sc_{xn}}{sc_x}
$$

$$
sc_{xn} = \frac{2^{PREC-1}-1}{\max{(|x|)}}
$$

#### Symmetric Channel-wise



#### Zero Point Layer-wise

Scale definition:
$$
sc_x = \frac{2^{PREC}-1}{\text{max}(x)-\text{min}(x)}
$$
Zero point definition:
$$
zp_x = sc_x \cdot \min{(x)}
$$
Float-to-int quantize formalization:
$$
Xq = \left\lfloor sc_x \cdot X-zp_x \right\rfloor
$$
Int-to-float re-quantize formalization:
$$
X = \frac{Xq + zp_x}{sc_x}
$$

Int-propagation quantize formalization:
$$
Xqn = \Bigg\lfloor \frac{sc_{xn}}{rsc_x}Xq + \frac{sc_{xn}}{rsc_x}rzp_x - zp_{xn}\Bigg\rfloor
$$

$$
Xqn \approx \text{cvm_shift}(frac_1 \cdot Xq, sb_1) + \text{cvm_shift}(frac_2, sb_2)
$$

where:
$$
frac_1, sb_1 = \text{cvm_float}\left(rsc_{xn}\right)
$$

$$
frac_2, sb_2 = \text{cvm_float}\left(rzp_{xn}\right)
$$

$$
rsc_{xn} = \frac{sc_{xn}}{rsc_x}
$$

$$
rzp_{xn} = rsc_{xn} \cdot rzp_x - zp_{xn}
$$

$$
sc_{xn} = \frac{2^{PREC-1}-1}{\max{(|x|)}}
$$

$$
zp_{xn} = sc_{xn} \cdot \min{(x)}
$$

#### Zero Point Channel-wise




### Multiplication Operations

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

2-D convolution formalization:
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
Y = \text{Convolution}(X, W, \text{stride}=\text{stride}, \text{dilation}=\text{dilation})
$$

$$
Y[n,o,p',q'] = \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} X[n,i,p',q'] \cdot W[o,i,ki,kj]
$$

where:
$$
p' = p \cdot SH + ki \cdot DH
$$

$$
q' = q \cdot SW + kj \cdot DW
$$

1. Symmetric Layer-wise quantized $X$ and $W$

$$
Y[n,o,p',q'] = \frac{1}{rsc_{xn} \cdot sc_w} \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqn[n,i,p',q'] \cdot Wq[o,i,ki,kj]
$$

Thus:
$$
Y[n,o,p,q] = \frac{1}{rsc_y} Yq[n,o,p,q]
$$
where:
$$
rsc_y = rsc_{xn} \cdot sc_w
$$

$$
Yq = \text{Convolution}(Xqn, Wq, \text{stride}=\text{stride}, \text{dilation}=\text{dilation})
$$

 $Xqn$ and $Wq$ is both of precision 8 of lower, i.e.
$$
\max{(|Xqn|)} \leq 2^{8-1} - 1
$$

$$
\max{(|Wq|)} \leq 2^{8-1} - 1
$$

Assert that:
$$
C \cdot KH \cdot KW <= 2^{16} - 1 = 65535
$$
which guarantees that:
$$
\max{(|Yq|)} \leq 2^{32-1} - 1
$$

2. Zero point Layer-wise quantized $X$ and $W$

$$
Y'[n,o,p',q'] = \frac{1}{rsc_{xn} \cdot sc_w} \Bigg\{ \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xqn[n,i,p',q'] \cdot Wq[o,i,ki,kj] \\
+ KH \cdot KW \cdot zp_w \sum_{i=0}^{C-1} Xqn[n,i,p',q'] 
+ zp_x \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
+ C \cdot KH \cdot KW \cdot zp_x \cdot zp_w \Bigg\}
$$

Thus:
$$
Y[n,o,p,q] = \frac{1}{rsc_y} \left(Yq[n,o,p,q] + rzp_y\right)
$$
where:
$$
rsc_y = rsc_{xn} \cdot sc_w
$$

$$
Yq = \text{Convolution}(Xqn, Wq, \text{stride}=\text{stride}, \text{dilation}=\text{dilation})
+ \text{sum}(Xqn, \text{keep_dims}=\text{True})
$$

$Xqn$ and $Wq$ is both of precision 8 of lower, i.e.
$$
\max{(|Xqn|)} \leq 2^{8} - 1
$$

$$
\max{(|Wq|)} \leq 2^{8} - 1
$$

Assert that:
$$
C \cdot KH \cdot KW <= 2^{16} - 1 = 65535
$$
which guarantees that:
$$
\max{(|Yq|)} \leq 2^{32-1} - 1
$$


### Addition Operations

