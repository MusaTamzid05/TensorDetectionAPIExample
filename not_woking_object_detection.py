import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from time import sleep


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from session_util import create_session


class ObjectDetector:

    def __init__(self,
            model_path,
            label_path):

        self.model_path = model_path
        self.path_to_frozen_graph = os.path.join(model_path , "frozen_inference_graph.pb")
        self.label_path = label_path

        print(self.model_path)
        print(self.path_to_frozen_graph)
        print(self.label_path)

        self.data_loaded = True


        if os.path.isdir(model_path) == False:
            print("does not exists =>{} .".format(model_path))
            self.data_loaded = False
            return

        if os.path.exists(label_path) == False:

            print("does not exists =>{} does not exists.".format(label_path))
            self.data_loaded = False
            return

        self._load_model()
        self._init_graph()

    def _load_model(self):

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_frozen_graph , "rb") as f:
                serilalized_graph = f.read()
                graph_def.ParseFromString(serilalized_graph)
                tf.import_graph_def(graph_def , name = "")

        print("Graph loaded in memory.")

    def _init_label_map(self):
        self.category_index = label_map_util.create_category_index_from_labelmap(self.label_path , use_display_name = True)

    def _init_graph(self):

        self.detection_graph= tf.Graph()

        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_frozen_graph , "rb") as f:
                serialized_graph = f.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def , name = "")


    def run_inference_for_single_image(self , sess ,  image , tensor_dict):

        if "detection_masks" in tensor_dict:
            tensor_dict["detection_masks"] = self._get_ditection_mask_reframed()

        image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")
        output_dict = sess.run(tensor_dict ,
                feed_dict = {
                    image_tensor : np.expand_dims(image , 0)
                    })

    def _get_ditection_mask_reframed(self , image , tensor_dict):

        detection_boxes = tf.squeeze(tensor_dict["detection_boxes"] , [0])
        detection_masks = tf.squeeze(tensor_dict["detection_masks"] , [0])
        real_num_detection = tf.cast(tensor_dict["num_detections"][0] , tf.int32)
        detection_boxes = tf.slice(detection_boxes , [0 , 0] , [real_num_detection , -1])

        detection_masks = tf.slice(detection_masks , [0,0,0] , [real_num_detection , -1 , -1])
        detection_masks_reframed = reframe_box_masks_to_image_masks(
                detection_masks,
                detection_boxes,
                image.shape[0],
                image.shape[1]
                )

        detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed , 0.5),
                tf.uint8
                )

        return  tf.expand_dims(detection_masks_reframed , 0)



    def load_tensor_dict(self):

        ops = tf.get_default_graph().get_operations()
        all_tensor_names = { output.name for op in ops for output in op.outputs }
        tensor_dict = {}

        for key in [
                "num_detections" ,
                "detection_boxes" ,
                "detection_scores",
                "detection_classes",
                "detection_masks"
                ]:
            tensor_name = key + ":0"

            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        return tensor_dict




    def detect_label_in(self , image):

        with create_session() as sess:
            tensor_dict = self.load_tensor_dict()
            print(tensor_dict)
            self.run_inference_for_single_image(sess , image , tensor_dict)
            print("Finished")







if __name__ == "__main__":

    object_detector = ObjectDetector(model_path = "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/result_graph" ,
            label_path = "/home/musa/custom_object_detector2/object_detection/models/research/object_detection/training/label_map.pbtxt")

    object_detector.detect_label_in("./Downloads/test.jpg")
