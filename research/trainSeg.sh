PIPELINE_CONFIG_PATH=/home/vtsai01/tf-models/research/ModaSeg/models/mask_rcnn_inception_resnet_v2_atrous_coco.config
TRAIN_DIR=/home/vtsai01/tf-models/research/ModaSeg/models/train/
NUM_TRAIN_STEPS=500000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 object_detection/model_main.py \
		    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
			--model_dir=${TRAIN_DIR} \
			--num_train_steps=${NUM_TRAIN_STEPS} \
			--alsologtostderr 

