# MRT Generalized Quantization Math Formalization

[TOC]

### Quantizer

|                | Parameter <br> (Uniform Symmetric Quantizer) | Input <br/> (Uniform Symmetric Quantizer)                    | Parameter <br/> (Uniform Affine Quantizer)                   | Input <br/> (Uniform Affine Quantizer)                       |
| -------------- | -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Scale          | $sc_{w} = \frac{2^{PREC-1}-1}{\max{|Wr|}}$   | $sc_{x} = \frac{2^{PREC-1}-1}{\max{|Xr|}}$                   | $sc_{w} = \frac{2^{PREC}-1}{\max{Wr} - \min{Wr}}$            | $sc_{x} = \frac{2^{PREC}-1}{\text{max}Xr-\text{min}Xr}$      |
| Zero Point     | -                                            | -                                                            | $zp_{wr} = \min Wr$                                          | $zp_{xe} = \text{round} \Big(\min Xr \cdot sc_{xe}\Big)$<br>$rzp_{x} = \text{round} \Big(\min Xr \Big)$ |
| Quantization   | $Wq = \text{round}\Big(sc_{w} \cdot Wr\Big)$ | $frac, exp = \text{cvm_float}\bigg(\frac{sc_{x}}{sc_{xe}}\bigg) \\ Xq = \text{realize} (X_{e}, frac, exp)$ | $W_{q} = \text{round} \Big[ (sc_{w} \left( W_{r} - zp_{wr} \right) \Big]$ | $frac, exp = \text{cvm_float}\bigg(\frac{sc_{x}}{sc_{xe}}\bigg) \\ Xq = \text{realize}(Xe - zp_{xe}, frac, exp)$ |
| Requantization | $Wr = \frac{Wq}{sc_{w}}$                     | $Xr = \frac{Xq}{sc_{x}}$                                     | $Wr = \frac{Wq}{sc_{w}} + zp_{wr}$                           | $Xr = \frac{Xq}{sc_{x}} + rzp_{x}$                           |

The variable whose names ending up with '**q**' stand for int quantized operators,  '**r**' stand for floating point operators and '**e**' stand for int expanded operators.

Re-quantization into expansion is elaborated in operator expansion, see [NN Operator Expansion](#nn-operator-expansion), [Broadcast Operator Expansion](#broadcast-operator-expansiono), [Elemwise Operator Expansion](#elemwise-operator-expansion), [Transform Operator Expansion](#transform-operator-expansion).

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

Take `Convolution` (for simplicity, only point-wise convolution is considered here, i.e. `num_group=1`) for instance (only uniform symmetric quantization is considered for simplicity), layer-wise Convolution can be rewritten as:
$$
\begin{align}

Ye[n,o,p,q] 

&= \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj] \\

&= \sum_{i=0}^{C/step-1} \sum_{j=i*step}^{(i+1)*step-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, j, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,j,ki,kj] \\

&= \sum_{i=0}^{C/step-1} Convolution(Xq[:,i*step:(i+1)*step,:,:], Wq[:,i*step:(i+1)*step,:,:]) \\

&= \sum_{i=0}^{C/step-1} Yei[n,o,p,q]

\end{align}
$$

#### Channel Merge

Merge the channel symbol components to the equivalent symbol. 

For operators like `pad`, `relu`, `Pooling`, merge as follows:
$$
X = \text{concat}\Big( \big[Xi, \forall i \in [0, C) \big], \text{dim=ich} \Big)
\tag{concat}
$$

For operators like `Convolution` (`num_group=1`), merge as follows:
$$
X = \sum_{i=0}^{C-1} Xi
\tag{add_n}
$$
For operators like `Convolution` (`num_group>1`), the slice channel process will be performed in each the output channel, and **concat** along the output channel axis.

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

Note, if `num_groups` is not 1, then convolution is generalized as **Groupwise Convolution**.

Specifically, suppose kernel weight $W$ is of shape $(O,IC,KH,KW)$ and input data $X$ is of shape $(N,C,H,W)$.
$$
Yr[n,o,p,q]

= \sum_{i=0}^{IC-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xr\Bigg[n, \bigg\lfloor\frac{o}{OPG}\bigg\rfloor IC + i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW\Bigg] \cdot Wr[o,i,ki,kj]
$$
For simplicity, here we will not inlcude the notation of groupwise convolution.

Given $Xe$ and $We$, MRT respectively quantize them into $Xq$ and $Wq$.

**Expansion Formalization 1: Symmetric Quantized X and W**
$$
Ye[n,o,p,q] = \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj]
$$

where the scale of $Ye$ is $sc_{x} \ sc_{w}$.

**Expansion Formalization 2: Zero Point Quantized X and Symmetric Quantized W**
$$
Ye1[n,o,p,q] = 
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj]
$$
$$
Ye2[n,o,p,q] = zp_{xe} \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Wq[o,i,ki,kj]
$$

where the scale of $Ye1$ is $sc_{x} \ sc_{w}$ and the scale of $Ye2$ is $sc_{w}$.  By [quantize_scale](#quantize-scale), MRT respectively quantize them into $Yq1$ and $Yq2$. Then get the final expansion.
$$
Ye = Yq1 + Yq2
$$
**Expansion Formalization 3: Symmetric Quantized X and Zero Point Quantized W**
$$
Ye1[n,o,p,q] =  
\sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW] \cdot Wq[o,i,ki,kj]
$$

$$
Ye2[n,o,p,q] =  
zp_{we} \sum_{i=0}^{C-1} \sum_{ki=0}^{KH-1} \sum_{kj=0}^{KW-1} Xq[n, i, p \cdot SH + ki \cdot DH, q \cdot SW + kj \cdot DW]
$$
where the scale of $Ye1$ is $sc_{x} \ sc_{w}$ and the scale of $Ye2$ is $sc_{x}$.  By [quantize_scale](#quantize-scale), MRT respectively quantize them into $Yq1$ and $Yq2$. Then get the final expansion.
$$
Ye = Yq1 + Yq2
$$
**Expansion Formalization 4: Zero Point Quantized X and W (Deprecated)**


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
Ye = quantize_scale(**Xqs, **attrs)
infer_prec = max(xprecs) if op_name == "Concat" else max(xprecs)+1
```

### Op-level Configuration Turorial

#### Optimized Quantization

MRT GEN support op-level **channel-slice** and **zero point specification** to optimize the quantization process for a better int inference result. Two steps are needed in the optimization process.

**Step 1. Find out potential layers to optimize**

A simple method is to print out the names of all the layers like follows. See [mrt.gen.Transformer.py](https://github.com/CortexFoundation/cvm-runtime/blob/archRev/python/mrt/gen/transformer.py)

```python
        print out zero point quantization candidates
        from mrt.sym_utils import is_params, sym_iter
        sym, params = self.current_model.symbol, self.current_model.params
        for s in topo_sort(sym):
            name, op_name = s.attr('name'), s.attr('op_name')
            if op_name in ["broadcast_add"]:
                childs = sym_iter(s.get_children())
                for c in childs:
                    cname = c.attr('name')
                    if is_params(c, params):
                        weight = params[cname]
                        maxv = weight.max().asscalar()
                        minv = weight.min().asscalar()
                        print(cname)
            elif op_name in ["Convolution"]:
                childs = sym_iter(s.get_children())
                for c in childs:
                    cname = c.attr('name')
                    if is_params(c, params):
                        weight = params[cname]
                        maxv = weight.max().asscalar()
                        minv = weight.min().asscalar()
                        print(maxv, minv, cname)
        exit()
```

**Step 2. Set up Cfg_groups in configuration file**

See [Configuration Example](https://github.com/CortexFoundation/cvm-runtime/blob/archRev/python/mrt/gen/model_zoo/config.example.ini). An examplary configuration is provide.

```ini
...
[CALIBRATION]
# [Optional] Calibration batch size, 16 by default.
Batch=16
# [Optional] Iterator numbers of calibration, 1 by default.
Calibrate_num=1
# [Optional] Granularity, Quantizer and Optimizor Configuration for specified nodes 
Cfg_groups=
  ["mrt_sym_separate_bias_alexnet0_conv0_fwd_0"]: gn_info: {"gn_type"; "channel-wise". "ichannel"; 1. "step"; 1},
  ["alexnet0_conv0_weight"]: quant_type: UniformAffine,
 	["mrt_sym_separate_bias_alexnet0_dense1_bias_0"]: quant_type: UniformAffine
...
```

#### Layer-wise Restoration

Also, both MRT and MRT GEN support layer-wise **restoration** for debugging purpose if MRT is developed for enhancement purposes. See [Configuration Example](https://github.com/CortexFoundation/cvm-runtime/blob/archRev/python/mrt/gen/model_zoo/config.example.ini). The restore can be specified by symbol names in `Restore_name`:

```ini
[QUANTIZATION]
...
# [Optional] Debug usage
Restore_name=
...
```

### Model Test

The comparison between the original float model, mrt quantized model and non-tuned mrt gen quantized model is listed as below.

| model name                    | Original Float Model          | MRT Quantized Model (Tuned)  | MRT Quantized Model           | MRT GEN Quanzied Model                             |
| ----------------------------- | ----------------------------- | ---------------------------- | ----------------------------- | -------------------------------------------------- |
| resnet50_v1                   | top1=77.39%<br />top5=93.59%  | top1=76.47%<br />top5=93.28% | top1=76.41%<br />top5=93.18%  | top1=75.66%<br />top2=92.79%                       |
| resnet50_v2                   | top1=77.15%<br />top5=93.44%  | top1=70.76%<br />top5=89.56% | top1=69.89%<br />top5=88.84%  | top1=69.83%<br />top5=88.84%                       |
| resnet18_v1                   | top1=70.96%<br />top5=89.93%  | top1=70.11%<br />top5=89.60% | top1=69.90%<br />top5=89.50%  | top1=69.97%<br />top5=88.84%                       |
| resnet18v1_b_0.89             | top1=67.21%<br />top5=87.45%  | -                            | top1=63.75%<br />top5=85.67%  | top1=69.97%<br />top5=89.53%                       |
| quickdraw                     | top1=81.90%<br />top5=98.26%  | -                            | top1=81.83%<br />top5=98.24%  | top1=81.89%<br />top5=98.22%                       |
| qd10_resnetv1_20              | top1=85.79%<br />top5=98.73%  | -                            | top1=85.79%<br />top5=98.73%  | top1=85.94%<br />top5=98.73%                       |
| densenet161                   | top1=77.62%<br />top5=93.82%  | -                            | top1=77.32%<br />top5=93.63%  | top1=76.90%<br />top5=93.49%                       |
| alexnet                       | top1=55.91%<br />top5=78.75%  | -                            | top1=51.69%<br />top5=77.99%  | top1=51.82%<br />top5=78.09%<br />(channel sliced) |
| cifar_resnet20_v1             | top1=92.88%<br />top5=99.78%  | -                            | top1=92.82%<br />top5=99.75%  | top1=92.52%<br />top5=99.79%                       |
| mobilenet1_0                  | top1=70.77%<br />top5=89.97%  | top1=66.11%<br />top5=87.35% | top1=63.07%<br />top5=85.02%  | top1=61.30%<br />top5=83.88%                       |
| mobilenetv2_1.0               | top1=71.51%<br />top5=90.10%  | top1=69.39%<br />top5=89.30% | top1=66.93%<br />top5=87.39%  | top1=62.15%<br />top5=84.00%                       |
| shufflenet_v1                 | top1=63.48%<br />top5=85.12%  | top1=60.45%<br />top5=82.95% | top1=60.40%<br />top5=82.91%  | top1=60.13%<br />top5=82.70%                       |
| squeezenet1.0                 | top1=57.20%<br />top5=80.04%  | top1=55.16%<br />top5=78.67% | top1=52.46%<br />top5=77.10%  | top1=54.36%<br />top5=78.19%                       |
| tf_inception_v3               | top1=55.58%<br />top5=77.56%  | top1=55.54%<br />top5=83.03% | top1=53.79%<br />top5=75.99%  | top1=52.13%<br />top5=73.96%                       |
| vgg19                         | top1=74.14%<br />top5=91.78%  | top1=73.75%<br />top5=91.67% | top1=73.70%<br />top5=91.66%  | top1=73.58%<br />top5=91.60%                       |
| mnist                         | top1=99.18%<br />top5=100.00% | -                            | top1=99.17%<br />top5=100.00% | top1=98.16%<br />top5=100.00%                      |
| trec                          | 97.84%                        | 97.63%                       | 97.20%                        | 97.38%                                             |
| yolo3_darknet53_voc           | 81.37%                        | -                            | 82.08%                        | 80.85%                                             |
| yolo3_mobilenet1.0_voc        | 75.98%                        | -                            | 71.53%                        | 70.70%                                             |
| ssd_512_resnet50_v1_voc       | 80.27%                        | -                            | 80.01%                        | 79.57%                                             |
| ssd_512_mobilenet1.0_voc      | 75.57%                        | 71.32%                       | 70.21%                        | 69.68%                                             |
| tf_mobilenet_v1_0.25_224_lite | top1=34.68%<br />top5=59.32%  | -                            | top1=3.39%<br />top5=9.93%    | top1=31.71%<br />top5=55.83%<br />(slice channel)  |

The MRT GEN module apply pad separate for `Convolution` and bias `separate` for both `Convolution` and `FullyConnected`. From the chart above we can that some model like `quickdraw`,  `qd10_resnetv1_20`, `resnet18v1_b_0.89`, `trec`, `squeezenet1.0` and  `alexnet` and have observable better accuracy than MRT module.
