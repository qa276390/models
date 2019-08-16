import time
import sys
import json
import numpy as np
import tensorflow as tf
import cv2 as cv
import imutils
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import os
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='prediction')
parser.add_argument('--model-path', type=str, default="my_exported_graphs-411163/frozen_inference_graph.pb", help='model path')
parser.add_argument('--pbtxt-path', type=str, default="ModalNetDetect/data/modalnet_label_map.pbtxt", help='pbtxt path')
parser.add_argument('--info-path', type=str, help='cloth info path')
parser.add_argument('--file-list', action='store_true', default=False, help='info file has only file list')
parser.add_argument('--img-dir', type=str, default="/eds/research/bhsin/yahoo_clothes/img/", help='img dir')
parser.add_argument('--output-path', type=str, help='output path')
parser.add_argument('--main-only', action='store_true', default=False, help='main part only')
parser.add_argument('--crop-all', action='store_true', default=False, help='crop all options')
parser.add_argument('--no-cut', action='store_true', default=False, help='not to cut off 20% in tops')
parser.add_argument('--prob-thr', type=float, default=0.5, help='img dir')

parser.add_argument('--gid2class', action='store_true', default=False, help='do we have gid2class')
parser.add_argument('--gid2class-path', type=str, help='gid2class path')




os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ep = 1e-9
ind2cat = { 1 : 'bags', 2 : 'accessories', 3 : 'shoes', 4 : 'shoes', 5 : 'outerwear', 6 : 'all-body', 7 : 'sunglasses', 8 : 'bottoms', 9 : 'tops', 10 : 'bottoms', 11: 'bottoms', 12 : 'hats', 13 : 'scarves'}
class Bbox:
    def __init__(self, xmin, ymin, xmax, ymax, score, class_id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.class_id = class_id
        self.area = (xmax-xmin)*(ymax-ymin)
        if(self.area<=0):
            print('error: area <= 0')
    def isSimilar(self, bbox):
        if(bbox.class_id==self.class_id):
            return True
        interarea = (min(self.xmax, bbox.xmax) - max(self.xmin, bbox.xmin)) * (min(self.ymax, bbox.ymax) - max(self.ymin, bbox.ymin))
        if(interarea/self.area > 0.9 or interarea/bbox.area > 0.9):
            return True
        else:
            return False
    def isSmaller(self, bbox):
        if(self.area<bbox.area):
            return True
        else:
            return False
class Ginfo:
    def __init__(self, gid, url, desc, cl1, cl2, cl3, cl4, clas):
        self.gid = gid
        self.url = url
        self.desc = desc
        self.cl1 = cl1
        self.cl2 = cl2
        self.cl3 = cl3
        self.cl4 = cl4
        self.clas = clas

def iscloth(ginfo):
    cl1 = ginfo.cl1
    cl2 = ginfo.cl2
    cl3 = ginfo.cl3
    cl4 = ginfo.cl4
    if(not cl1==23 and not cl1==42):
        return False
    elif(cl1==23 and cl2==220 and cl3==4451):
        return False
    elif(cl1==23 and cl2==392 and cl3==6740):
        return False
    elif(cl1==23 and cl2==552 and cl3==13698):
        return False
    elif(cl1==23 and cl2==738 and cl3==13222 and cl4==123190):
        return False
    elif(cl1==23 and cl2==780 and (cl3==14719 or cl3==14720)):
        return False
    else:
        return True


def isfashion(ginfo):
    cl1 = ginfo.cl1
    cl2 = ginfo.cl2
    cl3 = ginfo.cl3
    cl4 = ginfo.cl4
    if(cl1==9 or cl1==23 or cl1==25 or cl1==41 or cl1==42):
        return True
    else:
        return False

def draw_bbox_and_crop(args, sess, cropped_dir, testimg, graph_def, category_index, ginfo, setdata, metadata, main_part_only, indexofimage):
    #with tf.Session() as sess:
    if(True):
        if(main_part_only or args.crop_all):
            THR = args.prob_thr
            dect_num = 0
        else:
            THR = 0.40
            dect_num = 1

        # Read and preprocess an image.
        print('img_path:'+testimg)
        img = cv.imread(testimg)
        try:
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        except:
            print('imread error: not such file.')
            return False
        print('imread successed.')
        start_time = time.time()
        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        print("model predict: --- %s seconds ---" % round(time.time() - start_time, 2))
        #sess.close()
        boxlist = []
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        #print(num_detections)
        valid = False

        #check all the detections
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            dup = False
            if (classId==3):
                classId = 4
            if score > THR:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                ceil = bbox[2] * rows
                bnow = Bbox(x, y, right, ceil, score, classId)
                
                #check if the class match if we have gid2class
                if(args.gid2class):
                    detclass = ind2cat[bnow.class_id]
                    #detection error, not the class
                    if(not detclass == ginfo.clas):
                        continue

                #check if dup
                for btmp in boxlist:
                    if(main_part_only):
                        #should be shoes
                        if(ginfo.cl1==9 and not bnow.class_id==4):
                            dup = True
                            break
                        #should be bags
                        if((ginfo.cl1==25 or ginfo.cl1==41) and not bnow.class_id==1):
                            dup = True
                            break
                        if(bnow.isSmaller(btmp)):
                            dup = True
                            break
                        else:
                            boxlist.pop(0)
                    else:
                        if(bnow.isSimilar(btmp)):
                            dup = True
                            break

                        #gid2class[bnow.gid]
                if(dup):
                    #print('invalid')
                    continue
                boxlist.append(bnow)
        valid_detections = len(boxlist)
        print('val detections:'+str(valid_detections))

        if(valid_detections>dect_num):
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
                    dy = ceil - y
                    if(not args.no_cut):
                        #tops or outer: cut 20% off from bottom of img
                        if(classId==5 or classId==9):
                            imgcrop = img[int(y):int(ceil-0.2*dy), int(x):int(right)]
                        #bottoms: cut 20% off from top of img
                        elif(classId==8 or classId==10 or classId==11):
                            imgcrop = img[int(y+0.2*dy):int(ceil), int(x):int(right)]
                        else:
                            imgcrop = img[int(y):int(ceil), int(x):int(right)]
                    else:
                        imgcrop = img[int(y):int(ceil), int(x):int(right)]\
                    
                    new_image_id = ginfo.gid+'_'+str(i+1)

                    if(args.crop_all):
                        new_image_id = ginfo.gid+'-'+str(indexofimage)+'_'+str(i+1)

                    img_save = os.path.join(cropped_dir, new_image_id+'.jpg')
                    print('saved: ' + img_save)
                    cv.imwrite(img_save, imgcrop)
                    print(ind2cat[classId], "-->", score, x, y)
                    boxlist.append(bnow)
                    #print(category_index)
                    metadata[new_image_id] = {
                        'url_name' : ginfo.url,
                        'description' : ginfo.desc,
                        'categories' : [],
                        'title' : ginfo.desc,
                        'related' : [],
                        'category_id' : ginfo.cl1,
                        'semantic_category' : ind2cat[classId],
                        'category_id2' : ginfo.cl2,
                        'category_id3' : ginfo.cl3,
                        'category_id4' : ginfo.cl4,
                        'score' : score }
                        
                    aset['items'].append({
                        'item_id':ginfo.gid + '_' + str(i+1),
                        'index' : str(i+1)
                    }) 
            aset['set_id'] = ginfo.gid
        ######################### visualize ###########################

        #vis_util.visualize_boxes_and_labels_on_image_array(img, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=4, min_score_thresh=0.3) 
        #vis_util.visualize_boxes_and_labels_on_image_array(img, np.squeeze(out[2]), np.squeeze(out[3]).astype(np.int32), np.squeeze(out[1]), category_index, use_normalized_coordinates=True, line_thickness=4, min_score_thresh=THR)
        
        
        ######################### #########  ###########################

    return valid    
    #cv.imwrite('predict_result.jpg', img)
import copy
import codecs
def main():
    args = parser.parse_args()

    model_path = args.model_path
    pbtxt_path = args.pbtxt_path
    clothinfo_path = args.info_path
    img_dir = args.img_dir
    outpath = args.output_path
    main_part_only = args.main_only
    gid2class_path = args.gid2class_path
    cropped_dir = os.path.join(outpath, 'cropped_img')
    outdata_path = os.path.join(outpath, 'set_data')
    outmeta_path = os.path.join(outpath, 'meta_data')
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir) 

    label_map = label_map_util.load_labelmap(pbtxt_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    dfgid2class = pd.read_csv(gid2class_path)
    
    # Read the graph.
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    setdata = []
    metadata = {}
    count = 0
    notfound = 0
    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        with codecs.open(clothinfo_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                #print(line)
                cline = copy.deepcopy(line)
                spline = cline.split()
                n = len(spline)
                if(n<=0):
                    break
                try:
                    if(args.file_list):
                        gid = spline[0]
                        url = ""
                        desc = ""
                        cl4 = 0
                        cl3 = 0
                        cl2 = 0
                        cl1 = 0
                    else:
                        gid = spline[0]
                        url = spline[1]
                        desclist = spline[2:-4]
                        cl4 = int(spline[-1]) 
                        cl3 = int(spline[-2]) 
                        cl2 = int(spline[-3])
                        cl1 = int(spline[-4])
                        sep = ', '
                        desc = sep.join(desclist).replace(' ', '_')
                except:
                    print('format error: skip this line')
                    continue
                """
                print('-'*50)
                print(gid)
                print(url)
                print(desc)
                print(cl1)
                print(cl2)
                print(cl3)
                print(cl4)
                """ 
                

                ginfo = Ginfo(gid, url, desc, cl1, cl2, cl3, cl4, 0) #tmp
                
                if(iscloth(ginfo) and args.gid2class):
                    try:
                        clas = dfgid2class[dfgid2class.GD_ID==int(gid)].p_category.values[0]
                    except:
                        notfound+=1
                        print('gid:'+gid+' not found in gid2class table')
                        ### continue to next gid
                        continue
                else:
                    clas = " "

                ginfo = Ginfo(gid, url, desc, cl1, cl2, cl3, cl4, clas)
                if(args.file_list):
                    testimg = os.path.join(img_dir, gid)
                else:
                    testimg = os.path.join(img_dir, gid+'.jpg')
                
                #crop all                
                if(args.crop_all):
                    index = 0
                    testimg = os.path.join(img_dir, gid+'-'+str(index)+'.png')
                    try:
                        while(os.path.isfile(testimg)):
                            start_time = time.time()
                            valid = draw_bbox_and_crop(args, sess, cropped_dir, testimg, graph_def, category_index, ginfo, setdata, metadata, main_part_only, index)
                            print("draw and crop: --- %s seconds ---" % round(time.time() - start_time, 2))
                            testimg = os.path.join(img_dir, gid+'-'+str(index)+'.png')
                            index+=1
                    except:
                        print('path problem:',testimg)
                if(not isfashion(ginfo)):
                    continue
                #is watch
                if(cl1==25 and cl2==646 and cl3==11376 and cl4==125747):
                    continue
                if(not main_part_only and not iscloth(ginfo)):
                    continue

                start_time = time.time()
                valid = draw_bbox_and_crop(args, sess, cropped_dir, testimg, graph_def, category_index, ginfo, setdata, metadata, main_part_only, 0)
                print("draw and crop: --- %s seconds ---" % round(time.time() - start_time, 2))
                
                if(valid):
                    count+=1
                print('# of valid set: ' +str(count))
                if(count % 1000 == 0):
                    print('saving json file...')
                    with open(outdata_path+ "-" + str(count) + ".json", 'w', encoding = 'utf-8') as setfile:
                        json.dump(setdata, setfile, indent = 4)
                    with open(outmeta_path+ "-" + str(count) + ".json", 'w', encoding = 'utf-8') as metafile:
                        json.dump(metadata, metafile, indent=4, ensure_ascii=False)
            with open(outdata_path + ".json", 'w', encoding = 'utf-8') as setfile:
                json.dump(setdata, setfile, indent = 4)
            with open(outmeta_path + ".json", 'w', encoding = 'utf-8') as metafile:
                json.dump(metadata, metafile, indent=4, ensure_ascii=False) 
    
if __name__ == "__main__":
    main()
