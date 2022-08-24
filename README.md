# MORE

This repository is the official implementation of Multi-Head Model for Continual Learning via Out-of-Distribution Replay (MORE)

****** IMPORTANT ******
Please download the pre-trained transformer network from

https://drive.google.com/file/d/1uEpqe6xo--8jdpOgR_YX3JqTHqj34oX6/view?usp=sharing

and save the file as ./deit_pretrained/best_checkpoint.pth



# Requirements
Please install a virtual environment

```
conda create -n mcil python=3.8 anaconda
```

Activate the environment

```
conda activate mcil
```

Please install the following packages in the environment

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install ftfy
pip install timm==0.4.12
```

# Training
e.g., For CIFAR10-5T, train the network
```
python run.py --model deitadapter_more --n_tasks 5 --dataset cifar10 --adapter_latent 64 --optim sgd --compute_md --compute_auc --buffer_size 200 --n_epochs 20 --lr 0.005 --batch_size 64 --use_buffer --class_order 0 --folder cifar10
```

train the classifier (back-update)
```
python run.py --model deitadapter_more --n_tasks 5 --dataset cifar10 --adapter_latent 64 --optim sgd --compute_auc --buffer_size 200 --folder cifar10 --load_dir logs/cifar10 --n_epochs 10 --print_filename train_clf_epoch=10.txt --use_buffer --load_task_id 4 --train_clf --train_clf_save_name model_task_clf_epoch=10 --class_order 0
```
	
Change [--n_tasks --dataset --adapter_latent --buffer_size --folder --load_dir --n_epochs --lr] according to the values specified in the main paper to reproduce other experiments
	
# Testing
e.g., For CIFAR10-5T,
if back-update is used,
```
python run.py --model deitadapter_more --n_tasks 5 --dataset cifar10 --adapter_latent 64 --optim sgd --use_md --compute_auc --buffer_size 200 --folder cifar10 --load_dir logs/cifar10 --print_filename testing_train_clf_useMD.txt --use_buffer --load_task_id 4 --test_model_name model_task_clf_epoch=10_ --class_order 0
```
		
if back-update is not used.
```
python run.py --model deitadapter_more --n_tasks 5 --dataset cifar10 --adapter_latent 64 --optim sgd --use_md --compute_auc --buffer_size 200 --folder cifar10 --load_dir logs/cifar10 --print_filename testing_useMD.txt --use_buffer --load_task_id 4 --class_order 0
```

# Acknowledgement
The code format follows DER++, HAT

https://github.com/aimagelab/mammoth

https://github.com/joansj/hat


