# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

# kd
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1

# shake
CUDA_VISIBLE_DEVICES=0  python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill shake --model_s wrn_16_2 -a 0 -b 1 --trial 0|tee ./save/logs/wrn_16_2-9.log
CUDA_VISIBLE_DEVICES=0  python train_student.py  --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill shake  --model_s ShuffleV1 -a 0 -b 1 --trial 0|tee ./save/logs/ShuffleV1-9.log
CUDA_VISIBLE_DEVICES=0  python train_student.py  --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill shake  --model_s resnet8x4 -a 0 -b 1 --trial 0|tee ./save/logs/resnet8x4-9.log
CUDA_VISIBLE_DEVICES=0  python train_student.py  --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill shake   --model_s vgg8 -a 0 -b 1 --trial 0|tee ./save/logs/vgg8-9.log
CUDA_VISIBLE_DEVICES=0 python train_student.py  --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill shake    --model_s MobileNetV2 -a 0 -b 1 --trial 0|tee ./save/logs/MobileNetV2-9.log
CUDA_VISIBLE_DEVICES=0 python train_student.py  --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill shake  --model_s resnet20 -a 0 -b 1 --trial 0|tee ./save/logs/resnet20-110-9.log
CUDA_VISIBLE_DEVICES=0 python train_student.py  --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill shake   --model_s resnet20 -a 0 -b 1 --trial 0|tee ./save/logs/resnet20-56-9.log
CUDA_VISIBLE_DEVICES=0 python train_student.py  --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill shake --model_s vgg8 -a 0 -b 1 --trial 0|tee ./save/logs/vgg8-9.log