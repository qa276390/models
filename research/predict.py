import time
import copy
import codecs
import sys
import json
import numpy as np
import tensorflow as tf
import cv2 
import imutils
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import os
import argparse
import pandas as pd
from tqdm import tqdm


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
parser.add_argument('--debug', action='store_true', default=False, help='show debug msg')




os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ep = 1e-9
ind2cat = { 1 : 'bags', 2 : 'accessories', 3 : 'shoes', 4 : 'shoes', 5 : 'outerwear', 6 : 'all-body', 7 : 'sunglasses', 8 : 'bottoms', 9 : 'tops', 10 : 'bottoms', 11: 'bottoms', 12 : 'hats', 13 : 'scarves'}
cat2ind = {v: k for k, v in ind2cat.items()}
fuzzind = { 'bags' : 1, 'accessories' : 2, 'shoes' : 3, 'outerwear' : 9, 'all-body' : 9, 'sunglasses' : 2, 'bottoms' : 8, 'tops' : 9, 'hats' : 12, 'scarves': 13} 
def fcompare_issame(class1, class2):
    return fuzzind[class1] == fuzzind[class2]

#to detect if there is a table in our image
class ImageTable(object):

    def __init__(self, Image):
        self.image = Image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    def HorizontalLineDetect(self):
        ret, thresh1 = cv2.threshold(self.gray, 240, 255, cv2.THRESH_BINARY)
        blur = cv2.medianBlur(thresh1, 3)  # 
        blur = cv2.medianBlur(blur, 3)  # 
        h, w = self.gray.shape
        horizontal_lines = []
        for i in range(h - 1):
            if abs(np.mean(blur[i, :]) - np.mean(blur[i + 1, :])) > 90:
                horizontal_lines.append([0, i, w, i])
                cv2.line(self.image, (0, i), (w, i), (0, 255, 0), 2)

        horizontal_lines = horizontal_lines[1:]
        return horizontal_lines
    def VerticalLineDetect(self):
        edges = cv2.Canny(self.gray, 30, 240)
        minLineLength = 500
        maxLineGap = 30
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap).tolist()
        sorted_lines = sorted(lines, key=lambda x: x[0])

        vertical_lines = []
        for line in sorted_lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    vertical_lines.append((x1, y1, x2, y2))
                    cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return vertical_lines

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
        return self.isIntersect(bbox)
    def isIntersect(self, bbox):
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
        if args.debug:
            print('img_path:'+testimg)
        img = cv2.imread(testimg)
        try:
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv2.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        except:
            if args.debug:
                print('imread error: not such file.')
            return False
        if args.debug:
            print('imread successed.')
        start_time = time.time()
        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        if args.debug:
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
                    #if not the same class -> detection error, not the class
                    if(not fcompare_issame(detclass, ginfo.clas)):
                        if args.debug:
                            print('detect class:', detclass) 
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
                    elif args.crop_all: 
                        if bnow.isIntersect(btmp):
                            dup = True
                            break

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
        if(args.debug):
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
                    
                    if args.crop_all:
                        # Table Detection
                        _img = copy.deepcopy(imgcrop)
                        imageTBL = ImageTable(_img)
                        hor = imageTBL.HorizontalLineDetect()
                        ver = imageTBL.VerticalLineDetect()
                        nline = len(hor)+len(ver)
                        if nline >= 2:
                            if(args.debug):
                                print('Table Detected')
                            continue
                    
                    new_image_id = ginfo.gid+'_'+str(i+1)

                    if(args.crop_all):
                        new_image_id = ginfo.gid+'-'+str(indexofimage)+'_'+str(i+1)

                    img_save = os.path.join(cropped_dir, new_image_id+'.jpg')

                    cv2.imwrite(img_save, imgcrop)
                    if args.debug:
                        print(ind2cat[classId], "-->", score, x, y)
                        print('saved: ' + img_save)
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

    return valid    

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
    if args.gid2class:
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
            for line in tqdm(fp):
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
                    if args.debug:
                        print('format error: skip this line')
                    continue
                
                ginfo = Ginfo(gid, url, desc, cl1, cl2, cl3, cl4, 0) #tmp Ginfo, just for iscloth
                
                if(iscloth(ginfo) and args.gid2class):
                    try:
                        clas = dfgid2class[dfgid2class.GD_ID==int(gid)].p_category.values[0]
                    except:
                        notfound+=1
                        if args.debug:
                            print('gid:'+gid+' not found in gid2class table')
                        ### continue to next gid
                        continue
                else:
                    clas = " "

                ginfo = Ginfo(gid, url, desc, cl1, cl2, cl3, cl4, clas)
                if(args.file_list):
                    testimg = os.path.join(img_dir, gid)
                #crop all                
                elif(args.crop_all):
                    index = 0
                    testimg = os.path.join(img_dir, gid+'-'+str(index)+'.png')
                    try:
                        while(os.path.isfile(testimg)):
                            start_time = time.time()
                            valid = draw_bbox_and_crop(args, sess, cropped_dir, testimg, graph_def, category_index, ginfo, setdata, metadata, main_part_only, index)
                            if(args.debug):
                                print("draw and crop: --- %s seconds ---" % round(time.time() - start_time, 2))
                            index+=1
                            testimg = os.path.join(img_dir, gid+'-'+str(index)+'.png')

                            if(valid):
                                count+=1
                    except:
                        if args.debug:
                            print('path problem:',testimg)
                else:
                    testimg = os.path.join(img_dir, gid+'.jpg')
                    if(not isfashion(ginfo)):
                        continue
                    #is watch
                    if(cl1==25 and cl2==646 and cl3==11376 and cl4==125747):
                        continue
                    if(not main_part_only and not iscloth(ginfo)):
                        continue

                    start_time = time.time()
                    valid = draw_bbox_and_crop(args, sess, cropped_dir, testimg, graph_def, category_index, ginfo, setdata, metadata, main_part_only, 0)
                    if(args.debug):
                        print("draw and crop: --- %s seconds ---" % round(time.time() - start_time, 2))
                    
                    if(valid):
                        count+=1

                if(count % 1000 == 0):
                    print('# of valid set: ' +str(count))
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
