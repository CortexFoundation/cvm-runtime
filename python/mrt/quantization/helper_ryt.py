import mxnet as mx
from mxnet import ndarray as nd
from os import path
import tfm_pass as tpass
import sym_utils as sutils

import os
from os import path
import sys

lambd = 0.95
mu = 0.3
stride = 1

def get_conv_names(modelname):
    conv_name_dct_old = {}

    weight_thresh = {}
    myfile = path.expanduser('~/tvm-cvm/cvm/models/'+modelname+'_mid.ini')
    for k, v in [v.split(':') for v in load_file(myfile)]:
        wname = k.strip()
        thresh = float(v.replace(',', '').strip())
        weight_thresh[wname] = thresh

    myfile = path.expanduser('~/tvm-cvm/cvm/models/'+modelname+'_normal.ini')
    weight_normal = [v.strip() for v in load_file(myfile)]

    sym_file = path.expanduser('~/tvm-cvm/data/'+modelname+'.prepare.json')
    params_file = path.expanduser('~/tvm-cvm/data/'+modelname+'.prepare.params')
    sym = mx.sym.load(sym_file)
    params = nd.load(params_file)
    weight_2_conv = []
    for sym in sutils.topo_sort(sym, params):
        if sym.attr('op_name') == 'Convolution':
            name = sym.attr('name')
            wname = sutils.sym_iter(sym.get_children())[1].attr('name')
            if wname in weight_thresh or wname in weight_normal:
                continue
            weight_2_conv.append((wname, name))
    return weight_2_conv

def run_model(modelname):
    os.system('python cvm/quantization/main2.py cvm/models/'+modelname+'_auto.ini')
    exit()
    myfile = '/home/test/tvm-cvm/data/th.txt'
    content = load_file(myfile)
    w, th = [v.strip() for v in content[-1].split(',')]
    th = float(th)
    myfile = '/home/test/tvm-cvm/data/acc.txt'
    content = load_file(myfile)
    p0, p = [float(v.strip()) for v in content[-1].split(',')]
    return w, th, p0, p

def load_file(filepath):
    with open(filepath, 'r') as f:
        s = f.readlines()
    return s

def write_file(filepath, content):
    with open(filepath, 'w') as f:
        for line in content:
            f.write(line)

def combine_file(modelname):
    pfx = path.expanduser('~/tvm-cvm/cvm/models/'+modelname)
    top = load_file(pfx+'_top.ini')
    mid = load_file(pfx+'_mid.ini')
    base = load_file(pfx+'_base.ini')
    content = top+mid+base
    auto_file = pfx+'_auto.ini'
    write_file(auto_file, content)

def model_tuning(modelname):
    while True:
        w2c = get_conv_names(modelname)
        if not w2c:
            print('finished')
            break
        wname, conv_name = w2c[0]
        conv_restore_names = [name+'\n' for _, name in w2c[1:]]
        filepath = path.expanduser("~/tvm-cvm/data/conv_restore_names.txt")
        write_file(filepath, conv_restore_names)
        w, th, p0, p = run_model(modelname)
        print('juding: ', w, th, p0, p)
        if p0*lambd > p:
            m_th = math.ceil(mu*th)
            c_th = m_th
            max_p = p
            max_th = None
            while c_th>0:
                myfile = path.expanduser('~/tvm-cvm/cvm/models/'+modelname+'_mid.ini')
                content = load_file(myfile)
                pwname = content[-1].split(':')[0].strip()
                nline = '  ' + wname + ': ' + str(c_th)
                if not content or pwname != wname:
                    content[-1] += ','
                else:
                    content.pop()
                content.append(nline)
                write_file(myfile, content)
                _, _, _, p = run_model(modelname)
                if p>max_p:
                    max_th = c_th
                    max_p = p
                c_th -= stride
            if max_p < p0*lambd:
                assert False
            print(conv_name, wname, ': tuned', max_th)
        else:
            myfile = path.expanduser('~/tvm-cvm/cvm/models/'+modelname+'_normal.ini')
            wnames = load_file(myfile) +[wname+'\n']
            write_file(myfile, wnames)
            print(conv_name, wname, ': directly passed')

if __name__ == '__main__':
    assert len(sys.argv) == 2
    modelname = sys.argv[1]
    model_tuning(modelname)
