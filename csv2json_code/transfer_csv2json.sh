#bin/bash

. ./CONFIG

python3 csv2json.py \
    --image_dir ${TRAIN_IMAGE_DIR} \
    --categories_csv_path ${CATEGORIES_CSV_PATH} \
    --categories_json_path ${CATEGORIES_JSON_PATH} \
    --annotation_csv_path ${TRAIN_ANNOTATION_CSV_PATH} \
    --json_save_path ${TRAIN_JSON_SAVE_PATH}


python3 csv2json.py \
    --image_dir ${VAL_IMAGE_DIR} \
    --categories_csv_path ${CATEGORIES_CSV_PATH} \
    --categories_json_path ${CATEGORIES_JSON_PATH} \
    --annotation_csv_path ${VAL_ANNOTATION_CSV_PATH} \
    --json_save_path ${VAL_JSON_SAVE_PATH}


python3 csv2json.py \
    --image_dir ${TEST_IMAGE_DIR} \
    --categories_csv_path ${CATEGORIES_CSV_PATH} \
    --categories_json_path ${CATEGORIES_JSON_PATH} \
    --json_save_path ${TEST_JSON_SAVE_PATH}
