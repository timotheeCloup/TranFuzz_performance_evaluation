# madry robust training
# python3 defense/robust_training.py --dataset_target webcam --epochs 1 --model_target densenet --batch_size 8 --adv_trainer madry

# Tranfuzz defending
python3 defense/robust_TranFuzz.py --dataset_target webcam --epochs 1 --model_target densenet --batch_size 8 --clean_training True