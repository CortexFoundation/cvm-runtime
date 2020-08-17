# MRT Generalized Quantization Math Formalization

[TOC]

### Quantization

#### Symmetric Layer-wise

Scale Definition:
$$
sc_x = \frac{2^{PREC-1} - 1}{\max{(|Xr|)}}
$$

MRT symmetric symbol quantize process:
$$
Xq = \text{round}\Bigg( \frac{sc_x}{sc_{xe}} \Bigg) \cdot Xe
$$

MRT symmetric parameter quantize process:
$$
Wq = \text{round}(sc_w \cdot Wr)
$$

#### Symmetric Channel-wise



#### Zero Point Layer-wise

Scale definition:
$$
sc_x = \frac{2^{PREC}-1}{\text{max}(Xr)-\text{min}(Xr)}
$$
Zero point definition:
$$
zp_x = \big\lceil -sc_x \cdot \min{(Xr)} \big\rceil
$$
MRT zero point symbol quantize process:
$$
Xq = \text{round}\Bigg( \frac{sc_x}{sc_{xe}}\Bigg) \cdot Xe + zp_x
$$

MRT zero point parameter quantize process:
$$
Wq = \text{round}\Big( sc_w \cdot Wr \Big) + zp_w
$$

#### Zero Point Channel-wise




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

2-D convolution formalization:
$$
Y = \text{Convolution}(X, W, \text{stride}=\text{stride}, \text{dilation}=\text{dilation})
$$
Adequately:
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

1. Symmetric Layer-wise quantized $X$ and $W$

$$
Yr[n,o,p',q']
= \frac{1}{sc_x \cdot sc_w} \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n,i,p',q'] \cdot Wq[o,i,ki,kj]
$$

MRT expansion process:
$$
Ye = \text{Convolution}(Xq, Wq, \text{stride}=\text{stride}, \text{dilation}=\text{dilation})
$$

$$
sc_{ye} = sc_x \cdot sc_w
$$

Denote $Xq$ is of precision $xp$ and $Wq$ is of precision $wp$:
$$
\max{(|Xq|)} \leq 2^{xp-1} - 1
$$

$$
\max{(|Wq|)} \leq 2^{wp-1} - 1
$$

$$
\text{infer_prec} = \lceil \log{(C \cdot KH \cdot KW)} \rceil + xp + wp
$$

2. Zero point Layer-wise quantized $X$ and $W$

Since:
$$
Yr[n,o,p',q']
= \frac{1}{sc_x \cdot sc_w} \Bigg\{ \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n,i,p',q'] \cdot Wq[o,i,ki,kj] \\
- KH \cdot KW \cdot zp_w \sum_{i=0}^{C-1} Xq[n,i,p',q'] 
- zp_x \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
+ C \cdot KH \cdot KW \cdot zp_x \cdot zp_w \Bigg\}
$$
MRT expansion process:
$$
Ye = \text{Convolution}(Xq, Wq, \text{sride=stride}, \text{dilation=dilation}) 
+ C_1 \cdot \text{sum}(Xq, \text{axis=1}, \text{keep_dims=True}) 
+ C2
$$

$$
sc_{ye} = sc_x \cdot sc_w
$$

where:
$$
C_1 = -KH \cdot KW \cdot zp_w
$$

$$
C_2 = -zp_x \cdot \Big{[}C_1 + \text{sum} \left(Wq, \text{axis=(1,2,3)}, \text{keep_dims=True}\right) \Big{]}
$$

Denote $Xq$ is of precision $xp$ and $Wq$ is of precision $wp$:
$$
\max{(Xq)} \leq 2^{xp} - 1
$$

$$
\max{(Wq)} \leq 2^{wp} - 1
$$

$$
\text{infer_prec} = 2 + \max{ \Big{\{}
\big\lceil \log{(C \cdot KH \cdot KW)} \rceil + xp + wp + 2, \\
\big\lceil \log{(C \cdot KH \cdot KW) + \log{|zp_w|}} \big\rceil + xp + 1, \\
\big\lceil \log{(C \cdot KH \cdot KW) + \log{|zp_x|}} \big\rceil + wp + 1, \\
\big\lceil \log{(C \cdot KH \cdot KW) + \log{|zp_x|} + \log{|zp_w|}} \big\rceil
\Big{\}}}
$$

3. Zero point Layer-wise quantized $X$ and Symmetric Layer-wise quantized $W$

Since:
$$
Yr[n,o,p',q'] = \frac{1}{sc_x \cdot sc_w} \Bigg\{ 
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n,i,p',q'] \cdot Wq[o,i,ki,kj]
- zp_x \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
 \Bigg\}
$$
MRT expansion process:
$$
Ye = \text{Convolution}(Xq, Wq, \text{sride=stride}, \text{dilation=dilation}) 
+ C_1
$$

$$
sc_{ye} = sc_x \cdot sc_w
$$

where:
$$
C_1 = -zp_x \cdot \text{sum} \left(Wq, \text{axis=(1,2,3)}, \text{keep_dims=True}\right)
$$
Denote $Xq$ is of precision $xp$ and $Wq$ is of precision $wp$:
$$
\max{(Xq)} \leq 2^{xp} - 1
$$

$$
\max{(|Wq|)} \leq 2^{wp-1} - 1
$$

$$
\text{infer_prec} = 1 + \max{ \Big{\{}
\big\lceil \log{(C \cdot KH \cdot KW)} \rceil + xp + wp + 1, \\
\big\lceil \log{(C \cdot KH \cdot KW) + \log{|zp_x|}} \big\rceil + wp
\Big{\}}}
$$

4. Symmetric Layer-wise quantized $X$ and Zero point Layer-wise quantized $W$

Since:
$$
Yr[n,o,p',q'] = \frac{1}{sc_x \cdot sc_w} \Bigg\{ 
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n,i,p',q'] \cdot Wq[o,i,ki,kj]
- KH \cdot KW \cdot zp_w \sum_{i=0}^{C-1} Xq[n,i,p',q'] 
 \Bigg\}
$$
MRT expansion process:
$$
Ye = \text{Convolution}(Xq, Wq, \text{sride=stride}, \text{dilation=dilation}) 
+ C_1 \cdot \text{sum}(Xq, \text{axis=1}, \text{keep_dims=True})
$$
where:
$$
C_1 = -KH \cdot KW \cdot zp_w
$$
Denote $Xq$ is of precision $xp$ and $Wq$ is of precision $wp$:
$$
\max{(|Xq|)} \leq 2^{xp-1} - 1
$$

$$
\max{(Wq)} \leq 2^{wp} - 1
$$

$$
\text{infer_prec} = 1 + \max{ \Big{\{}
\big\lceil \log{(C \cdot KH \cdot KW)} \rceil + xp + wp + 1, \\
\big\lceil \log{(C \cdot KH \cdot KW) + \log{|zp_w|}} \big\rceil + xp
\Big{\}}}
$$



#### Pad



#### relu

#### Pooling

#### FullyConnected

### Broadcast Operator Expansion

