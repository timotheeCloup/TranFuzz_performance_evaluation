# 1 Entrainement train target
python3 DSAN/train_target_model.py --model_name densenet --epochs 200 --data_path ./datasets/office_home/RealWorld/ --dataset_target RealWorld --num_classes 65 --batch_size 64 --save_path ./DSAN/models
python3 test/predictions.py --target_path ./DSAN/models/target_RealWorld_densenet.pt --target_dataset RealWorld
# 2 Entrainement train source
python3 DSAN/DSAN.py --target_model DSAN/models/target_RealWorld_densenet.pt --target_name RealWorld --source_name Product
# 3.5 Test de la target pour différentes attaques
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method fgsm
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method pgd
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method cw
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method st

# 3 Fuzzing de la target sur dataset source (Resnet)
python3 fuzz/fuzzer_main.py --input_data ./datasets/office_home/Product/test/ --output_dir ./fuzz/data/target_RealWorld_densenet/ --input_model ./model_resnet50_Product_RealWorld.pth
python3 test/predictions.py --target_path ./DSAN/models/target_RealWorld_densenet.pt --target_dataset RealWorld  --target_model densenet --fuzz True


############################################################################################################
# ROBUSTESSE
############################################################################################################

# 4 Entrainement d'une target robuste Madry 
python3 defense/robust_training.py --dataset_target RealWorld --epochs 20 --model_target densenet --batch_size 64 --adv_trainer madry
# 5 Test accuracy de la target robuste Madry
python3 test/predictions.py --target_path ./defense/models/office_home/target_RealWorld_densenet_adv_madry.pt --target_dataset RealWorld --target_model densenet

# 5.5 Attaques sur la target robuste Madry
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method fgsm --model_path defense/models/office_home/target_RealWorld_densenet_adv_madry.pt
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method pgd --model_path defense/models/office_home/target_RealWorld_densenet_adv_madry.pt
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method cw --model_path defense/models/office_home/target_RealWorld_densenet_adv_madry.pt
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method st --model_path defense/models/office_home/target_RealWorld_densenet_adv_madry.pt

# 6 Fuzzing de la target robuste Madry sur dataset source (Resnet)
# Création d'un resnet source
python3 DSAN/DSAN.py --target_model defense/models/office_home/target_RealWorld_densenet_adv_madry.pt --target_name RealWorld --source_name Product --robust madry
python3 fuzz/fuzzer_main.py --input_data ./datasets/office_home/Product/test/ --output_dir ./fuzz/data/target_RealWorld_densenet_adv_madry/ --input_model ./model_resnet50_Product_RealWorld_madry.pth
# Test accuracy de la target robuste Madry
python3 test/predictions.py --target_path ./defense/models/office_home/target_RealWorld_densenet_adv_madry.pt --target_dataset RealWorld --target_model densenet --fuzz True --robust madry


############################################################################################################
# TranFuzz Defending
############################################################################################################

# 7 Entrainement d'une target robuste avec TranFuzz
python3 defense/robust_TranFuzz.py --dataset_target RealWorld --epochs 200 --model_target densenet --batch_size 64 --clean_training True
# 8 Test accuracy de la target robuste avec TranFuzz
python3 test/predictions.py --target_path ./defense/models/office_home/densenet_RealWorld_TranFuzz_defended.pt --target_dataset RealWorld --target_model densenet

# 8.5 Attaques sur la target robuste avec TranFuzz
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method fgsm --model_path defense/models/office_home/densenet_RealWorld_TranFuzz_defended.pt
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method pgd --model_path defense/models/office_home/densenet_RealWorld_TranFuzz_defended.pt
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method cw --model_path defense/models/office_home/densenet_RealWorld_TranFuzz_defended.pt
python3 attacks/attacks.py --dataset_target RealWorld --model_target densenet --batch_size 8 --attack_method st --model_path defense/models/office_home/densenet_RealWorld_TranFuzz_defended.pt

# 9 Fuzzing de la target robuste avec TranFuzz sur dataset source (Resnet)
python3 DSAN/DSAN.py --target_model ./defense/models/office_home/densenet_RealWorld_TranFuzz_defended.pt --target_name RealWorld --source_name Product --robust TranFuzz
python3 fuzz/fuzzer_main.py --input_data ./datasets/office_home/Product/test/ --output_dir ./fuzz/data/target_RealWorld_densenet_adv_TranFuzz/ --input_model ./model_resnet50_Product_RealWorld_TranFuzz.pth

# 10 Test accuracy fuzzing de la target robuste avec TranFuzz
python3 test/predictions.py --target_path ./defense/models/office_home/densenet_RealWorld_TranFuzz_defended.pt --target_dataset RealWorld --target_model densenet --fuzz True --robust TranFuzz
