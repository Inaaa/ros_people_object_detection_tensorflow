# tvm, relay
import tvm
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
from tvm.contrib import graph_runtime



import time

import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from object_detection.utils import ops as utils_ops

import rospkg

class Detector(object):
    def __init__(self, \
        model_name='ssd_mobilenet_v1_coco_11_06_2017',\
        num_of_classes=90,\
        label_file='mscoco_label_map.pbtxt',\
        num_workers=-1
        ):

        super(Detector, self).__init__()
        # What model to download.
        self._model_name = model_name
        # ssd_inception_v2_coco_11_06_2017

        self._num_classes = num_of_classes

        self._detection_graph = None

        self._sess = None
        self._state = None

        self.category_index = None

        self._label_file = label_file

        self._num_workers = num_workers
        #print(time.time)

        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()

        self._tf_object_detection_path = \
            rospack.get_path('cob_people_object_detection_tensorflow') + \
            '/src/object_detection'

        self._path_to_ckpt = self._tf_object_detection_path + '/' + \
            self._model_name + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        self._path_to_labels = self._tf_object_detection_path + '/' + \
            'data/' + self._label_file

        # Prepare the model for detection
        self.prepare()
    def load_model(self):
        """
        Loads the detection model

        Args:

        Returns:

        """

        with self._detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                self.od_grapf = tf.import_graph_def(self.od_graph_def, name='')
                self.od_graph_def = tf_testing.ProcessGraphDefParam(self.od_graph_def)
                # Add shapes to the graph.
                with tf.Session() as sess:
                    self.od_graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')

        label_map = label_map_util.load_labelmap(self._path_to_labels)
        self.label_map= label_map
        categories = label_map_util.convert_label_map_to_categories(\
            label_map, max_num_classes=self._num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def prepare(self):
        """
        Prepares the model for detection

        Args:

        Returns:

        """

        self._detection_graph = tf.Graph()

        self.load_model()

        # Set the number of workers of TensorFlow
        if self._num_workers == -1:
            self._sess = tf.Session(graph=self._detection_graph)
        else:
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=self._num_workers,
                inter_op_parallelism_threads=self._num_workers,
            )

            self._sess = tf.Session(graph=self._detection_graph,
                                    config=session_conf)

    def detect(self,image):
        #self.model_name ='ssd_mobilenet_v1_coco_11_06_2017'

        #self.label_map = 'mscoco_label_map.pbtxt'


        target = 'llvm'
        target_host = 'llvm'
        layout = None
        ctx = tvm.cpu(0)

######################################################################

# Decode image
# ------------
# .. note::
#
#   tensorflow frontend import doesn't support preprocessing ops like JpegDecode.
#   JpegDecode is bypassed (just return source node).
#   Hence we supply decoded frame to TVM instead.
#


        x = np.array(image)

#####################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
        shape_dict = {'DecodeJpeg/contents': x.shape}
        dtype_dict = {'DecodeJpeg/contents': 'uint8'}
        mod, params = relay.frontend.from_tensorflow(self.od_graph_def,
                                     layout=layout,
                                     shape=shape_dict)

        print("Tensorflow protobuf imported to relay frontend.")

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with TVM runtime.

        with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod[mod.entry_func],
                             target=target,
                             target_host=target_host,
                             params=params)
# OPT_PASS_LEVEL = {
#     "SimplifyInference": 0,
#     "OpFusion": 1,
#     "FoldConstant": 2,
#     "CombineParallelConv2D": 3,
#     "FoldScaleAxis": 3,
#     "AlterOpLayout": 3,
#     "CanonicalizeOps": 3,
# }
######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.

#from tvm.contrib import graph_runtime
        dtype = 'uint8'
        m = graph_runtime.create(graph, lib, ctx)
# set inputs
        m.set_input('DecodeJpeg/contents', tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
# execute
        m.run()
# get outputs
        tvm_output = m.get_output(0, tvm.nd.empty(((1, 1008)), 'float32'))

######################################################################
# Process the output
# ------------------
# Process the model output to human readable text for InceptionV1.
        predictions = tvm_output.asnumpy()
        predictions = np.squeeze(predictions)

# Creates node ID --> English string lookup.
        node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path,
                            uid_lookup_path=label_path)