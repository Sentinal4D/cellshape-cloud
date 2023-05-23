for shape in "plane"
  do
    for num_features in 512
      do
        python main_pl.py --batch_size 8 --learning_rate_autoencoder 0.00001 --encoder 'foldingnet' --shape $shape --num_features ${num_features} --num_epochs_autoencoder 1 --k 16 --pretrained_path /mnt/nvme0n1/pretrained_shapenet/Reconstruct_foldingnet_${shape}/models/
    done
  done
