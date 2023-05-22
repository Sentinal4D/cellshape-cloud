for shape in "plane" "sphere" "gaussian"
  do
    for num_features in 512 128 64 32 16 8 4 2
      do
        python main_pl.py --encoder 'foldingnet' --shape $shape --num_features ${num_features} --num_epochs_autoencoder 1 --k 16 --pretrained_path /mnt/nvme0n1/pretrained_shapenet/Reconstruct_foldingnet_${shape}/models/
    done
  done
