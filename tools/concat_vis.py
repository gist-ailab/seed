import cv2
import os


img_list = os.listdir('output/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_bs2/inference/vis')

for img in img_list:
    
    img_rcnn = cv2.imread(os.path.join('output/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_bs2/inference/vis/', img))
    img_box = cv2.imread(os.path.join('output/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs2_e3_re_l1_0.5_0.5_4.0_e2_brmh_0.5_ef_nf4_br80_cw2/inference/vis/', img))
    img_mask = cv2.imread(os.path.join('output/COCO-InstanceSegmentation/mask_eee_rcnn_R_50_FPN_1x_bs2_e3_re_l1_0.5_0.5_0.5_e2/inference/vis/', img))
    
    if img_box is None or img_mask is None:
        continue
    img_width = img_rcnn.shape[1]
    img_concat = cv2.hconcat([img_rcnn[:,:int(img_width/2),:], img_box[:,:int(img_width/2),:], img_mask])
    # img = cv2.resize(img, (640, 480))
    cv2.imwrite(os.path.join('vis', img), img_concat)
