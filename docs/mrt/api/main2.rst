.. _mrt_main2_api:

*************
MRT main2 API
*************

The main stages of main2 include the following 6 stages.

**1. prepare**

The raw model need to be prepared in order to be compatible with the mrt pipeline:

::

  model = Model.load(sym_path, prm_path)
  model.prepare(set_batch(input_shape, 1))

**2. split model**

In a model split operation, a model can be split into top and base by specifying keys. 

::

  base, top = model.split(keys)

**3. calibration**

A mrt instance is created for calibration and quantization stage.

::

  mrt = model.get_mrt() if keys == '' else base.get_mrt()

The calibration can be executed by specifying the number of calibration, lambd and data.

::

  for i in range(calibrate_num):
      data, _ = data_iter_func()
      mrt.set_data(data)
      mrt.calibrate(lambd=lambd, ctx=ctx)

**4. quantization**

A mrt instance can perform the quantization process, the user can set up some predefined parameters for mrt if needed, such as input precision, output precision, softmax lambd, shift bits as well as threshold for a particular node, etc.

::

  mrt.set_input_prec(input_precision)
  mrt.set_output_prec(output_precision)
  mrt.set_softmax_lambd(softmax_lambd)
  mrt.set_shift_bits(shift_bits)
  mrt.set_threshold(name, threshold)

Then, the quantization process is performed as follows:

:: 

  mrt.quantize()

**5. merge model**

By specifying the base and top models along with corresponding node key maps, the user can create a model merger instance.

::

  model_merger = Model.merger(qmodel, top, mrt.get_maps())

By specifying callback merging function, the user can merge the top and base models, and get the ouput scales by configure oscale_maps.

::

  qmodel = model_merger.merge(callback=mergefunc)
  oscale_maps = _get_val(
      cfg, sec, 'Oscale_maps', dtype=PAIR(str_t, str_t))
  oscales = model_merger.get_output_scales(
      mrt_oscales, oscale_maps)

**6. evaluation**

Quantized model reduction and performance comparison are implemented in the evaluation stage:
::

  org_model = Model.load(sym_path, prm_path)
  graph = org_model.to_graph(ctx=ctx)
  dataset = ds.DS_REG[ds_name](set_batch(input_shape, batch))
  data_iter_func = dataset.iter_func()
  metric = dataset.metrics()

  ...

  split_batch = batch//ngpus
  rqmodel = reduce_graph(qmodel, {
      'data': set_batch(input_shape, split_batch)})
  qgraph = rqmodel.to_graph(ctx=ctx)
  qmetric = dataset.metrics()

  ...

  utils.multi_validate(evalfunc, data_iter_func, quantize,
                       iter_num=iter_num,
                       logger=logging.getLogger('mrt.validate'),
                       batch_size=batch)



**7. compilation**

Compilation stage include model conversion from mxnet to cvm, and model dump:
::

  qmodel.to_cvm(model_name_tfm, datadir=dump_dir,
      input_shape=set_batch(input_shape, batch),
      target=device_type, device_ids=device_ids)

as well as dump of sample data and ext files:
::

  dump_data = sim.load_real_data(
      dump_data.astype("float64"), 'data', mrt.get_inputs_ext())
  model_root = path.join(dump_dir, model_name_tfm)
  np.save(path.join(model_root, "data.npy"),
          dump_data.astype('int8').asnumpy())
  infos = {
      "inputs_ext": inputs_ext,
      "oscales": oscales,
      "input_shapes": input_shape,
  }
  sim.save_ext(path.join(model_root, "ext"), infos)


mrt.main2
__________
.. automodule:: mrt.main2

.. autofunction:: mrt.main2.set_batch
.. autofunction:: mrt.main2.batch_axis
.. autofunction:: mrt.main2._check
.. autofunction:: mrt.main2._get_path
.. autofunction:: mrt.main2._get_ctx
.. autofunction:: mrt.main2.ARRAY
.. autofunction:: mrt.main2.PAIR
.. autofunction:: mrt.main2._get_val
.. autofunction:: mrt.main2._cast_val
.. autofunction:: mrt.main2._load_fname
.. autofunction:: mrt.main2._checkpoint_exist
