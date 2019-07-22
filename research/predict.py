import sys
import json
import numpy as np
import tensorflow as tf
import cv2 as cv
import imutils
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class Bbox:
    def __init__(self, xmin, ymin, xmax, ymax, score, class_id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.class_id = class_id
    def isSimilar(self, bbox):
        if(abs(bbox.xmin-self.xmin)/self.xmin < 0.01 and abs(bbox.ymin-self.ymin)/self.ymin < 0.01 and abs(bbox.xmax-self.xmax)/self.xmax < 0.01 and abs(bbox.ymax-self.ymax)/self.ymax< 0.01):
            return True
        else:
            return False
class Ginfo:
    def __init__(self, gid, url, desc, cl1, cl2, cl3, cl4):
        self.gid = gid
        self.url = url
        self.desc = desc
        self.cl1 = cl1
        self.cl2 = cl2
        self.cl3 = cl3
        self.cl4 = cl4




THR = 0.33
def draw_bbox_and_crop(cropped_dir, testimg, graph_def, category_index, ginfo, setdata, metadata):
    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        print('img_path:'+testimg)
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


        boxlist = []
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        #print(num_detections)
        valid = False

        #check
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            dup = False
            if score > THR:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                ceil = bbox[2] * rows
                bnow = Bbox(x, y, right, ceil, score, classId)
                for btmp in boxlist:
                    if(bnow.isSimilar(btmp)):
                        dup = True
                        break
                if(dup or classId==2 or classId==7):
                    print('invalid')
                    continue
                boxlist.append(bnow)
        valid_detections = len(boxlist)
        print('val detections:'+str(valid_detections))

        if(valid_detections>1):
            valid = True
            aset = {}
            aset['items'] = []
            setdata.append(aset)        
            for i in range(valid_detections):
                tmpbox = boxlist[i]
                classId = tmpbox.class_id
                score = tmpbox.score
                #bbox = [float(v) for v in out[2][0][i]]
                dup = False

                if score > THR:
                    x = tmpbox.xmin
                    y = tmpbox.ymin
                    right = tmpbox.xmax
                    ceil = tmpbox.ymax
                    
                    imgcrop = img[int(y):int(ceil), int(x):int(right)]
                    img_save = cropped_dir +'/' + ginfo.gid+'_'+str(i)+'.jpg'
                    print(img_save)
                    cv.imwrite(img_save, imgcrop)
                    print(classId, "-->", score, x, y)
                    boxlist.append(bnow)
                    #print(category_index)
                    metadata[ginfo.gid+'_'+str(i)] = {
                        'url_name' : ginfo.url,
                        'description' : ginfo.desc,
                        'categories' : [],
                        'title' : ginfo.desc,
                        'related' : [],
                        'category_id' : ginfo.cl1,
                        'semantic_category' : category_index[classId]['name'],
                        'category_id2' : ginfo.cl2,
                        'category_id3' : ginfo.cl3,
                        'category_id4' : ginfo.cl4 }
                        
                    aset['items'].append({
                        'item_id':ginfo.gid + '_' + str(i),
                        'index' : str(i)
                    }) 
            aset['set_id'] = ginfo.gid
        ######################### visualize ###########################

        #vis_util.visualize_boxes_and_labels_on_image_array(img, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=4, min_score_thresh=0.3) 
        vis_util.visualize_boxes_and_labels_on_image_array(img, np.squeeze(out[2]), np.squeeze(out[3]).astype(np.int32), np.squeeze(out[1]), category_index, use_normalized_coordinates=True, line_thickness=4, min_score_thresh=THR)
        
        
        ######################### #########  ###########################

    return valid    
    #cv.imwrite('predict_result.jpg', img)
import copy
import codecs
def main():

    model_path = "my_exported_graphs-231484/frozen_inference_graph.pb"
    pbtxt_path = "ModalNetDetect/data/modalnet_label_map.pbtxt"
    #testimg = "test_img/1641094109.jpg"
    #testimg = "test_img/F6.jpg"
    clothinfo_path = "test_img/cloth.data"
    img_dir = '/eds/research/bhsin/yahoo_clothes/img/'
    outpath = './test_yahoo_cloth'
    cropped_dir = os.path.join(outpath, 'cropped_img')
    outdata_path = os.path.join(outpath, 'set_data.json')
    outmeta_path = os.path.join(outpath, 'meta_data.json')
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir) 

    label_map = label_map_util.load_labelmap(pbtxt_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # Read the graph.
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    setdata = []
    metadata = {}
    count = 0
    with codecs.open(clothinfo_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            cline = copy.deepcopy(line)
            spline = cline.split()
            n = len(spline)
            if(n<=0):
                break
            gid = spline[0]
            url = spline[1]
            desclist = spline[2:-4]
            cl4 = spline[-1] 
            cl3 = spline[-2] 
            cl2 = spline[-3] 
            cl1 = spline[-4]
            sep = ', '
            desc = sep.join(desclist).replace(' ', '_')
            """
            print(gid)
            print(url)
            print(desc)
            print(cl1)
            print(cl2)
            print(cl3)
            print(cl4)
            print('-'*50)
            """
            ginfo = Ginfo(gid, url, desc, cl1, cl2, cl3, cl4)
            testimg = os.path.join(img_dir, gid+'.jpg')

            valid = draw_bbox_and_crop(cropped_dir, testimg, graph_def, category_index, ginfo, setdata, metadata)
            if(valid):
                count+=1
            print('# of valid set: ' +str(count))
    with open(outdata_path, 'w', encoding = 'utf-8') as setfile:
        json.dump(setdata, setfile, indent = 4)
    with open(outmeta_path, 'w', encoding = 'utf-8') as metafile:
        json.dump(metadata, metafile, indent=4, ensure_ascii=False)
        #metafile.write(unicode(d))
    
if __name__ == "__main__":
    main()
