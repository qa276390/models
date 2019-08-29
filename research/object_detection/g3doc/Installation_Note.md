# Tensorflow Object Detection API Note
## Issue: No module named 'object_detection' 
1. in models\research directory run the following:
`
    python setup.py build
    `
    `python setup.py install`
2. go to model/research/slim and run the following:
`sudo pip install -e .`
## Issue: ImportError: No module named 'pycocotools'
https://github.com/matterport/Mask_RCNN/issues/6
```bash

git clone https://github.com/pdollar/coco.git

cd coco/PythonAPI
make
sudo make install
sudo python setup.py install

before doing above steps install cython
```

```bash
pip install -U scikit-image
pip install -U cython
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

## Issue: Python Indent error
:%s/\t/    /g

## Issue: NameError: name 'unicode' is not defined
Another problem related to Python3:
models/research/object_detection/utils/object_detection_evaluation.py", line 290, in evaluate
category_name = unicode(category_name, 'utf-8')
NameError: name 'unicode' is not defined

**replace this: category_name = unicode(category_name, 'utf-8')**
**to this: category_name = str(category_name, 'utf-8')**


## Issue: ImportError: cannot import name 'string_int_label_map_pb2'
```bash
# From tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.
```
```bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

## Issue: locale.Error: unsupported locale setting
https://github.com/tensorflow/tensorboard/issues/1323
`export LC_ALL=C`
### Annotation Format
x_min,y_min,x_max,y_max,class_id

### Labels
Each polygon (bounding box, segmentation mask) annotation is assigned to one of the following labels:

| Label | Description | Mapping to | Fine-Grained-categories |
| --- | --- |--- | --- |
| 1 | bag | bags | bag |
| 2 | belt | accessories | belt |
| 3 | boots | shoes | boots |
| 4 | footwear(3) | shoes | footwear |
| 5 | outer | outerwear | coat/jacket/suit/blazers/cardigan/sweater/Jumpsuits/Rompers/vest |
| 6 | dress | all-body | dress/t-shirt dress |
| 7 | sunglasses | sunglasses | sunglasses |
| 8 | pants | bottoms |pants/jeans/leggings |
| 9 | top | tops |top/blouse/t-shirt/shirt |
|10 | shorts | bottoms | shorts |
|11 | skirt | bottoms | skirt |
|12 | headwear | hats | headwear |
|13 | scarf & tie | scarves | scartf & tie |

50K
