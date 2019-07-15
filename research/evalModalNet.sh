PIPELINE_CONFIG_PATH=/home/vtsai01/tf-models/research/objectdetect/models/model/modalnet_ssd_mobilenet_v2_coco.config
CHECKPOINT_DIR=/home/vtsai01/tf-models/research/objectdetect/models/model/ssd_mobilenet_v2_coco
EVAL_DIR=/home/vtsai01/tf-models/research/objectdetect/models/model/eval/
python3 eval.py \
		    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
			--checkpoint_dir=${CHECKPOINT_DIR} \
			--eval_dir=${EVAL_DIR} \
			--alsologtostderr


