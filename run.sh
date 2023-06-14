for shape in "plane" "sphere" "gaussian"
do
  for num_features in 512 128 64 32
  do
    python main_pl.py --batch_size 8 --num_epochs_autoencoder 250 --learning_rate_autoencoder 0.00001 --encoder 'dgcnn' --shape ${shape} --num_features ${num_features} --num_epochs_autoencoder 1 --k 40 --pretrained_path /mnt/nvme0n1/pretrained_shapenet/Reconstruct_dgcnn_k40_/Reconstruct_dgcnn_k40_${shape}/models/ --dataset_type "ShapeNet" --cloud_dataset_path /mnt/nvme0n1/Datasets/
    done
done
