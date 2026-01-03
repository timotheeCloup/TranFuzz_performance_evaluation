# 1 Entrainement train target
python3 DSAN/train_target_model.py --model_name densenet --epochs 200 --data_path ./datasets/office31/webcam/ --dataset_target webcam --num_classes 31 --batch_size 64 --save_path ./DSAN/models
python3 test/predictions.py --target_path ./DSAN/models/target_webcam_densenet.pt --target_dataset webcam
# 2 Entrainement train source
python3 DSAN/DSAN.py --target_model DSAN/models/target_webcam_densenet.pt --target_name webcam --source_name amazon
# 3.5 Test de la target pour différentes attaques
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method fgsm
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method pgd
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method cw
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method st

# 3 Fuzzing de la target sur dataset source (Resnet)
python3 fuzz/fuzzer_main.py --input_data ./datasets/office31/amazon/test/ --output_dir ./fuzz/data/target_webcam_densenet/ --input_model ./model_resnet50_amazon_webcam.pth
python3 test/predictions.py --target_path ./DSAN/models/target_webcam_densenet.pt --target_dataset webcam  --target_model densenet --fuzz True


############################################################################################################
# ROBUSTESSE
############################################################################################################

# 4 Entrainement d'une target robuste Madry 
python3 defense/robust_training.py --dataset_target webcam --epochs 20 --model_target densenet --batch_size 64 --adv_trainer madry
# 5 Test accuracy de la target robuste Madry
python3 test/predictions.py --target_path ./defense/models/office31/target_webcam_densenet_adv_madry.pt --target_dataset webcam --target_model densenet

# 5.5 Attaques sur la target robuste Madry
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method fgsm --model_path defense/models/office31/target_webcam_densenet_adv_madry.pt
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method pgd --model_path defense/models/office31/target_webcam_densenet_adv_madry.pt
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method cw --model_path defense/models/office31/target_webcam_densenet_adv_madry.pt
python3 attacks/attacks.py --dataset_target webcam --model_target densenet --batch_size 64 --attack_method st --model_path defense/models/office31/target_webcam_densenet_adv_madry.pt

# 6 Fuzzing de la target robuste Madry sur dataset source (Resnet)
# Création d'un resnet source
python3 DSAN/DSAN.py --target_model defense/models/office31/target_webcam_densenet_adv_madry.pt --target_name webcam --source_name amazon --robust madry
python3 fuzz/fuzzer_main.py --input_data ./datasets/office31/amazon/test/ --output_dir ./fuzz/data/target_webcam_densenet_adv_madry/ --input_model ./model_resnet50_amazon_webcam_madry.pth
# Test accuracy de la target robuste Madry
python3 test/predictions.py --target_path ./defense/models/office31/target_webcam_densenet_adv_madry.pt --target_dataset webcam --target_model densenet --fuzz True --robust madry