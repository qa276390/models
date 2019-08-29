GD_LIST="Yahoo_Data/clean_gd_list.txt"
#gid list with format "GID\tURL\tDESCRIPTION\tCLASS_1\tCLASS_2\tCLASS_3\tCLASS_4\t"
OUT_DIR="ex_output"
MODEL_PATH="Yahoo_Data/my_exported_graphs-411163/frozen_inference_graph.pb" 
PBTXT_PATH="./ModalNetDetect/data/modalnet_label_map.pbtxt"
IMG_DIR="/eds/research/bhsin/yahoo_clothes/img/"

time PYTHONIOENCODING=utf-8 python3 predict.py --info-path=$GD_LIST \
		--output-path=$OUT_DIR \
			--model-path=$MODEL_PATH \
				--pbtxt-path=$PBTXT_PATH \
					--img-dir=$IMG_DIR





#time PYTHONIOENCODING=utf-8 python3 predict.py /eds/research/bhsin/yahoo_clothes/clean_gd_list.txt yahoo_cloth_area_clean_cut 

