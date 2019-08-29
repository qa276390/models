GD_LIST=$1
#gid list with format "GID\tURL\tDESCRIPTION\tCLASS_1\tCLASS_2\tCLASS_3\tCLASS_4\t"
OUT_DIR=$2
MODEL_PATH=$3
PBTXT_PATH=$4
IMG_DIR=$5

time PYTHONIOENCODING=utf-8 python3 predict.py --info-path="{$GD_LIST}" \
		--output-path="{$OUT_DIR}" \
			--model-path="{$MODEL_PATH}" \
				--pbtxt-path="{$PBTXT_PATH}" \
					--img-dir="{$IMG_DIR}"


#time PYTHONIOENCODING=utf-8 python3 predict.py /eds/research/bhsin/yahoo_clothes/clean_gd_list.txt yahoo_cloth_area_clean_cut 

