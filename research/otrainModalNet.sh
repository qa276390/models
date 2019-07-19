PIPELINE_CONFIG_PATH=/home/tsai/tf-models/research/ModalNetDetect/models/model/modalnet_ssd_mobilenet_v2_coco.config
#MODEL_DIR=/home/tsai/tf-models/research/ModalNetDetect/models/model/ssd_mobilenet_v2_coco
TRAIN_DIR=/home/tsai/tf-models/research/ModalNetDetect/models/model/train/
NUM_TRAIN_STEPS=500000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 object_detection/model_main.py \
		    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
			--model_dir=${TRAIN_DIR} \
			--num_train_steps=${NUM_TRAIN_STEPS} \
			--alsologtostderr 

