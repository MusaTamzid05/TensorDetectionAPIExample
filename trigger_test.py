import tensorflow as tf

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils.label_map_util import create_category_index_from_labelmap

import cv2
import argparse

from util import get_labels

'''
for video =>
 python detect_object_player.py --src "/home/musa/test.mp4" --train_model_path "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17" --label_map_path "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/data/mscoco_label_map.pbtxt"

for webcam =>
 python detect_object_player.py --src "0" --train_model_path "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17" --label_map_path "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/data/mscoco_label_map.pbtxt"


'''

def create_session(fraction = 0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def run_inference_for_single_image(sess, image, graph , tensor_dict):

    if 'detection_masks' in tensor_dict:

        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})


    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def detect_object_in( model_path , label_map_path , video_src):
    PATH_TO_FROZEN_GRAPH = model_path + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = label_map_path
    detection_graph = tf.Graph()

    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    with detection_graph.as_default():
        with create_session()  as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
                ]:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            cap = cv2.VideoCapture(video_src)
            video_running = True

            while True:
                ret , image_np = cap.read()

                output_dict = run_inference_for_single_image(sess,image_np, detection_graph, tensor_dict)
                detected_labels = get_labels(
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index
                        )

                print(detected_labels)
                vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=8)
                cv_image = cv2.resize(image_np, (800, 600))
                cv2.imshow("Press q to quit" , cv_image)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    break





def test():
    detect_object_in(
            model_path = "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/result3",
            label_map_path = "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/training/label_map.pbtxt",
            video_src = "/home/musa/test.mp4")


def test2():

    detect_object_in(
            model_path = "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17",
            label_map_path = "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/data/mscoco_label_map.pbtxt",
            video_src = "/home/musa/test.mp4")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_model_path', dest = "model_path" , required = True ,   type = str ,  help='the model path after generating inference graph')
    parser.add_argument('--label_map_path', dest = "label_map_path" , required = True ,   type = str ,  help='path to the label map')
    parser.add_argument('--src', dest = "src" ,  type = str, required = True ,   help='video src')
    args = parser.parse_args()

    video_src = args.src
    model_path = args.model_path
    label_map_path = args.label_map_path

    try:
        video_src = int(video_src)
    except ValueError:
        pass


    detect_object_in(
            model_path = model_path,
            label_map_path = label_map_path,
            video_src = video_src
            )


if __name__ == "__main__":
    test2()


