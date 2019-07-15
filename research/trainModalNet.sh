PIPELINE_CONFIG_PATH=/home/vtsai01/tf-models/research/objectdetect/models/model/modalnet_ssd_mobilenet_v2_coco.config
MODEL_DIR=/home/vtsai01/tf-models/research/objectdetect/models/model/ssd_mobilenet_v2_coco
NUM_TRAIN_STEPS=500000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 object_detection/model_main.py \
		    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
			--model_dir=${MODEL_DIR} \
			--num_train_steps=${NUM_TRAIN_STEPS} \
			--sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
			--alsologtostderr


