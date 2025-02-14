python scripts/ct_lipro_inference.py \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 10 \
    --warmup_length 10000 \
    --save /mnt/home/admvkl@median.cad/code/public/CT-CLIP/output \
    --pretrained /mnt/home/admvkl@median.cad/code/public/CT-RATE/models/CT-CLIP-Related/CT_LiPro_v2.pt \
    --data-folder /mnt/home/admvkl@median.cad/code/public/CT-RATE/dataset/valid \
    --reports-file /mnt/home/admvkl@median.cad/code/public/CT-RATE/dataset/radiology_text_reports/validation_reports.csv \
    --labels /mnt/home/admvkl@median.cad/code/public/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv