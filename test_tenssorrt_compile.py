import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import rewrite
from tvm.relay.dataflow_pattern import *
import numpy as np
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor 
import json
# import pygraphviz as pgv
from argparse import ArgumentParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *
from tvm import rpc
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
from tensorflow import keras
from tvm.contrib.download import download_testdata
# from tvm.contrib import relay_viz
from tvm.relay import build_module
from PIL import Image
import time
import os.path



from tensorflow.keras.applications.resnet50 import preprocess_input

# Data
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

data = np.array(img)[np.newaxis, :].astype("float32")
data = preprocess_input(data).transpose([0, 3, 1, 2])

precompiled_file_path = "./pre_compiled_resnet50.so"

if not os.path.isfile(precompiled_file_path):
    # ###########################################################################
    # import resnet50
    weights_url = "".join(
        [
            " https://storage.googleapis.com/tensorflow/keras-applications/",
            "resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet50_keras_new.h5"

    weights_path = download_testdata(weights_url, weights_file, module="keras")
    model_keras = tf.keras.applications.resnet.ResNet50(
        include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
    )
    model_keras.load_weights(weights_path)

################################################

    shape_dict = {"input_1": data.shape}
    mod, params = relay.frontend.from_keras(model_keras, shape_dict)
    from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
    mod = partition_for_tensorrt(mod, params)

    target = "cuda"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    lib.export_library(precompiled_file_path)
else:
    lib = tvm.runtime.load_module(precompiled_file_path)

################################################
# # Original
target = 'cuda'
dev = tvm.cuda(0)
# dev = tvm.device("cuda", 1)
# target = 'cuda -arch=sm_61'
# dev = tvm.cuda(1)
# mod = mod
# with tvm.transform.PassContext(opt_level=4):
#     lib = relay.build(mod, target, params=params)
total_model = graph_executor.GraphModule(lib["default"](dev))

for i in range(50):
    total_model.set_input('input_1', data)
    total_model.run()
    total_model.get_output(0).numpy()

now = time.time()
print("Running...")
for i in range(100):
    total_model.set_input('input_1', data)
    total_model.run()
    total_model.get_output(0).numpy()

# print("Single model 2080ti 100iter: ", time.time() - now)
print("Single model with tensorrt compiled nvidia-jetson-agx-xavier 100iter: ", time.time() - now)


# dtype = "float32"
# input_shape = (1, 3, 224, 224)
# block = get_model('resnet18_v1', pretrained=True)
# mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)

# from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
# mod, config = partition_for_tensorrt(mod, params)

# target = "cuda"
# with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
#     lib = relay.build(mod, target=target, params=params)

# lib.export_library('compiled.so')

# dev = tvm.cuda(0)
# loaded_lib = tvm.runtime.load_module('compiled.so')
# gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))
# input_data = np.random.uniform(0, 1, input_shape).astype(dtype)
# gen_module.run(data=input_data)