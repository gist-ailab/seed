export CUDA_VISIBLE_DEVICES=0

# Evaluate Mask R-CNN + SEED
python train_net.py --config-file configs/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs16_e3_re_l1_0.5_1.0_2.0_e2_brmh_0.3_ef_nf3_br80_cw2_topb_nfc0_nc3_1414_decafmf_ie128_bffc1.yaml --num-gpus 8 --dist-url tcp://127.0.0.1:50160 --resume --eval-only

