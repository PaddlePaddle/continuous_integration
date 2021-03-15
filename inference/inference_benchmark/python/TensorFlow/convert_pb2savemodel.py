import sys
import argparse
import logging

import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %v(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.info("==== Tensorflow version: {} ====".format(tf.version.VERSION))
logger.info("==== Tensorflow version less than 2.4 can execute this codes ====")


def parse_args():
    """
    parse input args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model .pb input path")
    parser.add_argument(
        "--output_path", type=str, help="tf saved model output path")
    parser.add_argument(
        "--input_node",
        type=str,
        default="input/input_data:0",
        help="tf model input node")
    parser.add_argument(
        "--output_node",
        type=str,
        default="pred_sbbox/concat_1:0",
        help="tf model output node")
    return parser.parse_args()


def pb_to_savedmodel(args):
    """
    convert pb stucture model to saved model
    """
    if not args.model_path:
        logger.error("==== no input model found ====")
        sys.exit(1)
    if not args.output_path:
        logger.error("==== no output_path found ====")
        sys.exit(1)

    export_dir = args.output_path
    graph_pb = args.model_path

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    sigs = {}
    logger.info("==== start saved_model ====")
    with tf.Session(graph=tf.Graph()) as sess:
        # name="" is important to ensure we don't get spurious prefixing
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        inp = g.get_tensor_by_name(args.input_node)
        out = g.get_tensor_by_name(args.output_node)

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"in": inp}, {"out": out})

        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING], signature_def_map=sigs)
    builder.save()
    logger.info("==== finish saved_model ====")


if __name__ == "__main__":
    args = parse_args()
    pb_to_savedmodel(args)
