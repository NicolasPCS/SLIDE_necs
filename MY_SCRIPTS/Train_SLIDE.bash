cd pointnet2

## OVER MIRRORED
## Train DDPMs that Learn the Positions of the Sparse Latent Points
#python train.py --config configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156.json -g shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane --dist_url tcp://127.0.0.1:29500
#python train.py --config configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_car_02958343.json -g shapenet_psr_keypoint_generation_batchsize_32_with_ema_car --dist_url tcp://127.0.0.1:29501
#python train.py --config configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_chair_03001627.json -g shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair --dist_url tcp://127.0.0.1:29502

## Train DDPMs that Learn the Features of the Sparse Latent Points
#python train_latent_ddpm.py --config configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json -g shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane --dist_url tcp://127.0.0.1:29503
#python train_latent_ddpm.py --config configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_car_ae_trained_on_car.json -g shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car --dist_url tcp://127.0.0.1:29504
#python train_latent_ddpm.py --config configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_chair_ae_trained_on_chair.json -g shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_chair_ae_trained_on_chair --dist_url tcp://127.0.0.1:29505

## OVER HALF
### Airplane

## Train Point Cloud Autoencoders

#python distributed.py --config configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_airplane.json

#python distributed.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_kp_latent_dim_16_32_normal_weight_0.1_with_augm_kp_noise_0.04_not_add_centroid_as_first_kp_airplane.json

## Train DDPMs that Learn the Positions of the Sparse Latent Points

#python train.py --config configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156.json -g shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane --dist_url tcp://127.0.0.1:29500

#python train.py --config configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156_not_add_centroid_as_first_kp.json -g shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane_not_add_centroid_as_first_keypoint --dist_url tcp://127.0.0.1:29501

## Train DDPMs that Learn the Features of the Sparse Latent Points

#python train_latent_ddpm.py --config configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json -g shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane --dist_url tcp://127.0.0.1:29500

#python train_latent_ddpm.py --config configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_kp_noise_0.04_keypoint_conditional_not_add_centroid_as_first_kp_airplane.json -g shapenet_psr_latent_ddpm_kp_noise_0.04_keypoint_conditional_airplane_not_add_centroid_as_first_keypoint --dist_url tcp://127.0.0.1:29501

### Car

## Train Point Cloud Autoencoders

#python distributed.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_car.json 

#python distributed.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_kp_latent_dim_16_32_normal_weight_0.1_with_augm_kp_noise_0.04_not_add_centroid_as_first_kp_car.json

## Train DDPMs that Learn the Positions of the Sparse Latent Points

#python train.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_car_02958343.json -g shapenet_psr_keypoint_generation_batchsize_32_with_ema_car --dist_url tcp://127.0.0.1:29500

#python train.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_car_02958343_not_add_centroid_as_first_kp.json -g shapenet_psr_keypoint_generation_batchsize_32_with_ema_car_not_add_centroid_as_first_keypoint --dist_url tcp://127.0.0.1:29501

## Train DDPMs that Learn the Features of the Sparse Latent Points

#python train_latent_ddpm.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_car_ae_trained_on_car.json -g shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car --dist_url tcp://127.0.0.1:29502

#python train_latent_ddpm.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_kp_noise_0.04_keypoint_conditional_not_add_centroid_as_first_kp_car.json -g shapenet_psr_latent_ddpm_kp_noise_0.04_keypoint_conditional_car_not_add_centroid_as_first_keypoint --dist_url tcp://127.0.0.1:29503

### Chair

## Train Point Cloud Autoencoders

#python distributed.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_chair.json

#python distributed.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_kp_latent_dim_16_32_normal_weight_0.1_with_augm_kp_noise_0.04_not_add_centroid_as_first_kp_chair.json

## Train DDPMs that Learn the Positions of the Sparse Latent Points

#python train.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_chair_03001627.json -g shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair --dist_url tcp://127.0.0.1:29500

#python train.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_chair_03001627_not_add_centroid_as_first_kp.json -g shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair_not_add_centroid_as_first_keypoint --dist_url tcp://127.0.0.1:29501

## Train DDPMs that Learn the Features of the Sparse Latent Points

#python train_latent_ddpm.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_chair_ae_trained_on_chair.json -g shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_chair_ae_trained_on_chair --dist_url tcp://127.0.0.1:29502

#python train_latent_ddpm.py --config /home/isipiran/SLIDE_necs/pointnet2/configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_kp_noise_0.04_keypoint_conditional_not_add_centroid_as_first_kp_chair.json -g shapenet_psr_latent_ddpm_kp_noise_0.04_keypoint_conditional_chair_not_add_centroid_as_first_keypoint --dist_url tcp://127.0.0.1:29503