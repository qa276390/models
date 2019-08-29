time PYTHONIOENCODING=utf-8 python3 predict.py --prob-thr 0.6 --no-cut \
	--info-path ./Yahoo_Data/clothes.data \
	--output-path crop_all_tbdetect \
	--img-dir /eds/research/bhsin/yahoo_clothes/clothes_full_image/ \
	--crop-all \
	--gid2class \
	--gid2class-path ./Yahoo_Data/gid2class.csv \

