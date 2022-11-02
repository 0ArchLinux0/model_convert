import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("./slim_160_latest.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("slim_160_latest.pb")