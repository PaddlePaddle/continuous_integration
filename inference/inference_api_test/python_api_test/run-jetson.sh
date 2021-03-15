project_path=$(cd "$(dirname "$0")";pwd)

mkdir -p Data
cd Data
wget --no-proxy https://sys-p0.bj.bcebos.com/inference_jetson/Data.zip
unzip Data.zip

cd ..

#declare -A ModelCase
ModelCase["gpu_jetson"]="bert_emb_v1-paddle.py \
	          electra-deploy.py \
                  blazeface_nas_128.py \
                  ch_rec_use_space_char.py \
                  deeplabv3_resnet50_os8_cityscapes_1024x512_80k.py \
                  deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.py \
                  det_mv3_east.py \
                  dete_dist_yolov3_v1_uncombined.py \
                  faster_rcnn_r50_1x.py \
                  faster_rcnn_r50_fpn_1x_coco.py \
                  fcn_hrnetw18_cityscapes_1024x512_80k.py \
                  fcn_hrnetw18_voc12aug_512x512_40k.py \
                  gcnet_resnet50_os8_voc12aug_512x512_40k.py \
                  hardnet_cityscapes_1024x1024_160k.py \
                  hub-ernie-model.py \
                  mask_rcnn_r50_1x.py \
                  MobileNetV1_pretrained.py \
                  ppyolo_r50vd_dcn_1x_coco.py \
                  pspnet_resnet50_os8_cityscapes_1024x512_80k.py \
                  rec_chinese_common_train.py \
                  rec_chinese_lite_train.py \
                  rec_mv3_none_bilstm_ctc.py \
                  rec_mv3_none_none_ctc.py \
                  resnet50.py \
                  ResNet50_pretrained.py \
                  SE_ResNeXt50_32x4d_pretrained.py \
                  slim_quan_v1_aware_combined.py \
                  ssd_vgg16_300_240e_voc.py \
                  ttfnet_darknet53_1x_coco.py \
                  unet_cityscapes_1024x512_160k.py \
                  Xception_41_pretrained.py \
                  yolov3_darknet.py \
                  yolov3_darknet53_270e_coco.py"



ModelCase["trt_jetson_fp16"]="bert_emb_v1-paddle.py \
                              electra-deploy.py \ 
	                      blazeface_nas_128.py \
                              ch_rec_use_space_char.py \
                              deeplabv3_resnet50_os8_cityscapes_1024x512_80k.py \
                              deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.py \
                              det_mv3_east.py \
                              dete_dist_yolov3_v1_uncombined.py \
                              faster_rcnn_r50_1x.py \
                              faster_rcnn_r50_fpn_1x_coco.py \
                              fcn_hrnetw18_cityscapes_1024x512_80k.py \
                              fcn_hrnetw18_voc12aug_512x512_40k.py \
                              gcnet_resnet50_os8_voc12aug_512x512_40k.py \
                              hardnet_cityscapes_1024x1024_160k.py \
                              hub-ernie-model.py \
                              mask_rcnn_r50_1x.py \
                              MobileNetV1_pretrained.py \
                              ppyolo_r50vd_dcn_1x_coco.py \
                              pspnet_resnet50_os8_cityscapes_1024x512_80k.py \
                              rec_chinese_common_train.py \
                              rec_chinese_lite_train.py \
                              rec_mv3_none_bilstm_ctc.py \
                              rec_mv3_none_none_ctc.py \
                              resnet50.py \
                              ResNet50_pretrained.py \
                              SE_ResNeXt50_32x4d_pretrained.py \
                              slim_quan_v1_aware_combined.py \
                              ssd_vgg16_300_240e_voc.py \
                              ttfnet_darknet53_1x_coco.py \
                              unet_cityscapes_1024x512_160k.py \
                              Xception_41_pretrained.py \
                              yolov3_darknet.py \
                              yolov3_darknet53_270e_coco.py"




export project_path
for config in "gpu_jetson" "trt_jetson_fp16" 
do
    cd ${project_path}/tests/${config}
    rm -rf log.txt
    touch log.txt
    for file in ${ModelCase[${config}]}
    do
        python3.6  ${file}
    done
done
