## Airplane

cd /home/isipiran/SLIDE_necs/pointnet2/sampling_and_inference

# Generate Positions of the Sparse Latent Points

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane/checkpoint/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir ../exps/generated_point_cloud_and_mesh/airplane/centroid

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane_not_add_centroid_as_first_keypoint/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156_not_add_centroid_as_first_kp.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane_not_add_centroid_as_first_keypoint/checkpoint/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir ../exps/generated_point_cloud_and_mesh/airplane/random

# Generate Features of the Sparse Latent Points and Reconstruct Point Clouds

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane/checkpoint/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps/generated_point_cloud_and_mesh/airplane/centroid/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps/generated_point_cloud_and_mesh/airplane/centroid

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_kp_noise_0.04_keypoint_conditional_airplane_not_add_centroid_as_first_keypoint/config_latent_ddpm_s3_kp_noise_0.04_keypoint_conditional_not_add_centroid_as_first_kp_airplane.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_kp_noise_0.04_keypoint_conditional_airplane_not_add_centroid_as_first_keypoint/checkpoint/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps/generated_point_cloud_and_mesh/airplane/random/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps/generated_point_cloud_and_mesh/airplane/random

## Car

# Generate Positions of the Sparse Latent Points

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_car/config_standard_attention_batchsize_32_s3_ema_model_keypoint_car_02958343.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_car/checkpoint/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir ../exps/generated_point_cloud_and_mesh/car/centroid

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_car_not_add_centroid_as_first_keypoint/config_standard_attention_batchsize_32_s3_ema_model_keypoint_car_02958343_not_add_centroid_as_first_kp.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_car_not_add_centroid_as_first_keypoint/checkpoint/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir ../exps/generated_point_cloud_and_mesh/car/random

# Generate Features of the Sparse Latent Points and Reconstruct Point Clouds

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_car_ae_trained_on_car.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car/checkpoint/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps/generated_point_cloud_and_mesh/car/centroid/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps/generated_point_cloud_and_mesh/car/centroid

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_kp_noise_0.04_keypoint_conditional_car_not_add_centroid_as_first_keypoint/config_latent_ddpm_s3_kp_noise_0.04_keypoint_conditional_not_add_centroid_as_first_kp_car.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_kp_noise_0.04_keypoint_conditional_car_not_add_centroid_as_first_keypoint/checkpoint/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps/generated_point_cloud_and_mesh/car/random/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps/generated_point_cloud_and_mesh/car/random

## Chair

# Generate Positions of the Sparse Latent Points

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair/config_standard_attention_batchsize_32_s3_ema_model_keypoint_chair_03001627.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair/checkpoint/pointnet_ckpt_1483999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir ../exps/generated_point_cloud_and_mesh/chair/centroid

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair_not_add_centroid_as_first_keypoint/config_standard_attention_batchsize_32_s3_ema_model_keypoint_chair_03001627_not_add_centroid_as_first_kp.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair_not_add_centroid_as_first_keypoint/checkpoint/pointnet_ckpt_1483999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir ../exps/generated_point_cloud_and_mesh/chair/random

# Generate Features of the Sparse Latent Points and Reconstruct Point Clouds

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_chair_ae_trained_on_chair/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_chair_ae_trained_on_chair.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_chair_ae_trained_on_chair/checkpoint/pointnet_ckpt_1483999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps/generated_point_cloud_and_mesh/chair/centroid/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps/generated_point_cloud_and_mesh/chair/centroid

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_kp_noise_0.04_keypoint_conditional_chair_not_add_centroid_as_first_keypoint/config_latent_ddpm_s3_kp_noise_0.04_keypoint_conditional_not_add_centroid_as_first_kp_chair.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_kp_noise_0.04_keypoint_conditional_chair_not_add_centroid_as_first_keypoint/checkpoint/pointnet_ckpt_1483999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps/generated_point_cloud_and_mesh/chair/random/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps/generated_point_cloud_and_mesh/chair/random

#### ORIGNAL ARCHITECTURE ASSESING
## Airplane
# Generate Positions of the Sparse Latent Points

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/add_centroid_to_first_keypoint/airplane/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/add_centroid_to_first_keypoint/airplane/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir /home/isipiran/SLIDE_necs/pointnet2/exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/airplane/centroid

# Generate Features of the Sparse Latent Points and Reconstruct Point Clouds

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/airplane/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/airplane/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/airplane/centroid/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/airplane/centroid

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/airplane/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/airplane/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/airplane/centroid/mirrored_shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/airplane/centroid/mirrored

## Car

# Generate Positions of the Sparse Latent Points

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/add_centroid_to_first_keypoint/car/config_standard_attention_batchsize_32_s3_ema_model_keypoint_car_02958343.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/add_centroid_to_first_keypoint/car/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir /home/isipiran/SLIDE_necs/pointnet2/exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/car/centroid

# Generate Features of the Sparse Latent Points and Reconstruct Point Clouds

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/car/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_car_ae_trained_on_car.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/car/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/car/centroid/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/car/centroid

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/car/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_car_ae_trained_on_car.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/car/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/car/centroid/mirrored_shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/car/centroid/mirrored

## Chair

# Generate Positions of the Sparse Latent Points

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/add_centroid_to_first_keypoint/chair/config_standard_attention_batchsize_32_s3_ema_model_keypoint_chair_03001627.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/add_centroid_to_first_keypoint/chair/pointnet_ckpt_1483999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir /home/isipiran/SLIDE_necs/pointnet2/exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/chair/centroid

# Generate Features of the Sparse Latent Points and Reconstruct Point Clouds

export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/chair/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_chair_ae_trained_on_chair.json \
--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/chair/pointnet_ckpt_1483999.pkl \
--ema_idx 0 --batch_size 100 \
--keypoint_file ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/chair/centroid/shapenet_psr_generated_data_16_pts.npz \
--save_dir ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/chair/centroid/centroid

export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/chair/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_chair_ae_trained_on_chair.json \
--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps_old/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/chair/pointnet_ckpt_1483999.pkl \
--ema_idx 0 --batch_size 100 \
--keypoint_file ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/chair/centroid/mirrored_shapenet_psr_generated_data_16_pts.npz \
--save_dir ../exps_architecture_assesing_only_gen/over_original_dataset/generated_point_cloud_and_mesh/chair/centroid/centroid/mirrored

#### MIRRORED ARCHITECTURE ASSESING
## Airplane
# Generate Positions of the Sparse Latent Points

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane/checkpoint/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir /home/isipiran/SLIDE_necs/pointnet2/exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/airplane/centroid

# Generate Features of the Sparse Latent Points and Reconstruct Point Clouds

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane/checkpoint/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/airplane/centroid/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/airplane/centroid

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane/checkpoint/pointnet_ckpt_884999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/airplane/centroid/mirrored_shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/airplane/centroid/mirrored

## Car

# Generate Positions of the Sparse Latent Points

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_car/config_standard_attention_batchsize_32_s3_ema_model_keypoint_car_02958343.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_car/checkpoint/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir /home/isipiran/SLIDE_necs/pointnet2/exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/car/centroid

# Generate Features of the Sparse Latent Points and Reconstruct Point Clouds

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_car_ae_trained_on_car.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car/checkpoint/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/car/centroid/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/car/centroid

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_car_ae_trained_on_car.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car/checkpoint/pointnet_ckpt_1639999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/car/centroid/mirrored_shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/car/centroid/mirrored

## Chair

# Generate Positions of the Sparse Latent Points

#export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair/config_standard_attention_batchsize_32_s3_ema_model_keypoint_chair_03001627.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair/checkpoint/pointnet_ckpt_1483999.pkl \
#--ema_idx 0 --num_samples 1000 --batch_size 200 \
#--save_dir /home/isipiran/SLIDE_necs/pointnet2/exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/chair/centroid

# Generate Features of the Sparse Latent Points and Reconstruct Point Clouds

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_chair_ae_trained_on_chair/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_chair_ae_trained_on_chair.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_chair_ae_trained_on_chair/checkpoint/pointnet_ckpt_1483999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/chair/centroid/shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/chair/centroid/centroid

#export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_chair_ae_trained_on_chair/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_chair_ae_trained_on_chair.json \
#--ckpt /home/isipiran/SLIDE_necs/pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_chair_ae_trained_on_chair/checkpoint/pointnet_ckpt_1483999.pkl \
#--ema_idx 0 --batch_size 100 \
#--keypoint_file ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/chair/centroid/mirrored_shapenet_psr_generated_data_16_pts.npz \
#--save_dir ../exps_architecture_assesing_only_gen/over_mirrored_dataset/generated_point_cloud_and_mesh/chair/centroid/centroid/mirrored