TRAIN_IMAGE_DIR=~/ModalNet_Dataset/train_images
VAL_IMAGE_DIR=~/ModalNet_Dataset/train_images
TEST_IMAGE_DIR=~/ModalNet_Dataset/train_images
TRAIN_ANNOTATIONS_FILE=~/ModalNet_Dataset/annotations/nzero_modanet2018_instances_train.json
VAL_ANNOTATIONS_FILE=~/ModalNet_Dataset/annotations/nzero_modanet2018_instances_val.json
TESTDEV_ANNOTATIONS_FILE=~/ModalNet_Dataset/annotations/nzero_modanet2018_instances_val.json
OUTPUT_DIR=./ModaSeg
python3 ./object_detection/dataset_tools/create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}" \
	  --include_masks true
