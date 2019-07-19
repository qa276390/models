CHECKPOINT_NUMBER=194219
MODEL_DIR=ModalNetDetect/models/model
CKPT_PREFIX=${MODEL_DIR}/train/model.ckpt-${CHECKPOINT_NUMBER}
PIPELINE_CONFIG_PATH=/home/tsai/tf-models/research/ModalNetDetect/models/model/modalnet_ssd_mobilenet_v2_coco.config

python3 object_detection/export_inference_graph.py \
		  --input_type image_tensor \
		    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
			  --trained_checkpoint_prefix ${CKPT_PREFIX} \
			    --output_directory my_exported_graphs
