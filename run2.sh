############################################################################################################
# TranFuzz Defending
############################################################################################################

# 7 Entrainement d'une target robuste avec TranFuzz
python3 defense/robust_TranFuzz.py --dataset_target webcam --epochs 200 --model_target densenet --batch_size 64 --clean_training True
# 8 Test accuracy de la target robuste avec TranFuzz
python3 test/predictions.py --target_path ./defense/models/office31/densenet_webcam_TranFuzz_defended.pt --target_dataset webcam --target_model densenet

# 8.5 Attaques sur la target robuste avec TranFuzz
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method fgsm --model_path defense/models/office31/densenet_webcam_TranFuzz_defended.pt
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method pgd --model_path defense/models/office31/densenet_webcam_TranFuzz_defended.pt
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method cw --model_path defense/models/office31/densenet_webcam_TranFuzz_defended.pt
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method st --model_path defense/models/office31/densenet_webcam_TranFuzz_defended.pt

# 9 Fuzzing de la target robuste avec TranFuzz sur dataset source (Resnet)
python3 DSAN/DSAN.py --target_model ./defense/models/office31/densenet_webcam_TranFuzz_defended.pt --target_name webcam --source_name amazon --robust TranFuzz
python3 fuzz/fuzzer_main.py --input_data ./datasets/office31/amazon/test/ --output_dir ./fuzz/data/target_webcam_densenet_adv_TranFuzz/ --input_model ./model_resnet50_amazon_webcam_TranFuzz.pth

# 10 Test accuracy fuzzing de la target robuste avec TranFuzz
python3 test/predictions.py --target_path ./defense/models/office31/densenet_webcam_TranFuzz_defended.pt --target_dataset webcam --target_model densenet --fuzz True --robust TranFuzz
