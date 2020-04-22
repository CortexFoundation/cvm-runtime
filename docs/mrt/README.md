## MRT Installation



#### 1. Conda Environment Configuration

1.1. Install conda if there is no previous installation

```
pip install conda
```

1.2. Set up local python `3.7` environment for conda

​	First, check out whether the enviroment exists or not:

```
conda info --envs
```

<200b>  If not, add source channels into `~/.condarc`, eg.

>channels:
>
>  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
>
>  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
>
>  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
>
>  - https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
>
>  - https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
>
>  - https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
>
>  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
>
>  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
>
>  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
>
>show_channel_urls: true

​	Then, create the environment named `py3`:

```
conda create -n py3 python=3.7
```

​	Then, switch to the environment.

```
conda activate py3
```

1.3. Check up whether the environment is configured successfully

```
which pip
```

```
which python
```



#### 2. MRT Preparation

2.1. Clone the project

```
cd ~
git clone --recursive https://github.com/CortexFoundation/tvm-cvm.git
```

2.2. Build the Shared Library

```
make -j8
```

2.3. Install other dependancies

​	First, open the file `requirements.txt` to see which dependancies are to be installed. 

​	The default file looks like this:

>Mxnet-cu92mkl
>
>gluoncv
>
>decorator

​	If other packages are also needed while testing, append the package name at a newline.

​	Use the following command to install all the dependancies.

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<200b>  An alternative command is available.

```
pip install -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/web/simple
```

2.4. Create a Data Folder

```
mkdir data
```



#### 3. Test Model

```
python cvm/quantization/test_cifar10_resnetv1_20.py
```

```bash
python cvm/quantization/test_resnet.py
```



## Reference

https://pypi.org/project/conda/#files

https://mxnet.apache.org/versions/master/install/index.html?platform=Linux&language=Python&processor=GPU

https://blog.csdn.net/qq_28193895/article/details/80705809

https://anaconda.org/conda-forge/xz
