time PYTHONIOENCODING=utf-8 python3 predict.py --prob-thr 0.8 --no-cut \
	--info-path /eds/research/bhsin/yahoo_clothes/clothes.data \
	--output-path crop_all \
	--img-dir /eds/research/bhsin/yahoo_clothes/clothes_full_image/ \
	--crop-all \
	--gid2class \
	--gid2class-path ./gid2class.csv

