import numpy as np
import tensorflow as tf
import cv2 as cv
import imutils
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

model_path = "my_exported_graphs-231484/frozen_inference_graph.pb"
pbtxt_path = "ModalNetDetect/data/modalnet_label_map.pbtxt"
testimg = "test_img/1641094109.jpg"

label_map = label_map_util.load_labelmap(pbtxt_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# Read the graph.
with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    img = cv.imread(testimg)
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    ######################### visualize ###########################
    
    #vis_util.visualize_boxes_and_labels_on_image_array(img, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=4, min_score_thresh=0.3)
    
    vis_util.visualize_boxes_and_labels_on_image_array(img, np.squeeze(out[2]), np.squeeze(out[3]).astype(np.int32), np.squeeze(out[1]), category_index, use_normalized_coordinates=True, line_thickness=4, min_score_thresh=0.3)
    
    
    ######################### #########  ###########################




    """
    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    print(num_detections)
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]

        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            print(classId, "-->", score, x, y)
    """ 
cv.imwrite('predict_result.jpg', img)
#cv.imshow('SHOW', imutils.resize(img, width=800))
#cv.waitKey()
