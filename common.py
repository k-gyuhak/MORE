from argparse import ArgumentParser
from utils.best_args import best_args

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--load_dir', type=str, default=None, help='if provided, load model and test')
    parser.add_argument('--load_task_id', type=int, default=None)
    parser.add_argument('--print_filename', type=str, default=None, help="if None, prints on 'result.txt' file")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['mnist', 'cifar100', 'cifar10', 'timgnet', 'imagenet'])
    parser.add_argument('--model', type=str, default='derpp', choices=['derpp', 'derpp_deit', 'joint', 'owm', 'birch', 'ood', 'hier',
                                                                        'hcluster', 'oe', 'oe_fixed_minibatch', 'maha', 'maha_oe',
                                                                        'batch_pca', 'batch_pca_task', 'batch_pca_single',
                                                                        'batch_pca_deit', 'batch_pca_task_deit', 'batch_pca_single_deit',
                                                                        'maha_ipca', 'maha_ipca_task', 'maha_ipca_single',
                                                                        'maha_ipca_deit', 'maha_ipca_task_deit' 'maha_ipca_single_deit',
                                                                        'agem_r', 'singleSigma', 'hal', 'moco_feature', 'vitadapter_more',
                                                                        'clipadapter', 'clipadapter_hat', 'clipadapter_amp',
                                                                        'clipadapter_hat_amp', 'derpp_vitadapter',
                                                                        'derpp_deitadapter', 'deitadapter_more',
                                                                        'pass_fixed_deit', 'pass_vitadapter', 'pass_deitadapter', 'pass_resnet18', 'pass_alexnet',
                                                                        'icarl_vitadapter', 'icarl_deitadapter',
                                                                        'agem_r_vitadapter', 'agem_r_deitadapter',
                                                                        'owm_vitadapter', 'owm_deitadapter',
                                                                        'hal_vitadapter', 'hal_deitadapter',
                                                                        'vitadapter_hat_amp'])
    parser.add_argument('--noCL', action='store_true')
    parser.add_argument('--task_type', type=str, default='standardCL_randomcls',
                            choices=['cov', 'concept', 'pre-define',
                                     'standardCL_supercls', 'standardCL_randomcls'], help='learning scenarios')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_task', type=int, default=0)
    parser.add_argument('--n_tasks', type=int, default=5)
    parser.add_argument('--validation', type=float, default=None, help='Propertion of dataset used e.g. if set 0.9, 90\% of training data is used for training and rest 10\% is used for validation')
    parser.add_argument('--optim_type', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--zero_shot', action='store_true', help='Print zeroshot accuracy')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--init_epoch', type=int, default=0, help='initial epoch. Epoch starts from init_epoch and finishes at n_epochs-1')
    parser.add_argument('--loss_f', type=str, default='ce', choices=['ce', 'bce', 'nll'])
    parser.add_argument('--revisit', type=int, default=2, help='number of revisits')
    parser.add_argument('--confusion', action='store_true')
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--prob', type=float, default=None, help='probability; how many samples of a class are used for revisit')
    parser.add_argument('--coin', type=int, default=None, choices=[0, 1], help='whether a class experiences concept shift or not')
    parser.add_argument('--choose', type=int, default=0, help='whether a class experiences concept shift or not')
    parser.add_argument('--clip_init', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--separate_buffer', action='store_true')
    parser.add_argument('--use_buffer', action='store_true', help='if true, use buffer. Some systems do not use buffer by default. Use it for them.')
    parser.add_argument('--epsilon', type=float, default=None, help='epsilon noise for ODIN')
    parser.add_argument('--T_odin', type=float, default=None, help='temperature scale for ODIN')
    parser.add_argument('--modify_previous_ood', action='store_true')
    parser.add_argument('--select', action='store_true', help='if true, update only the heads of classes in current batch, and fix other heads')
    parser.add_argument('--choice', default='uniform')
    parser.add_argument('--holdout', type=int, default=None, help='number of holdout samples per class. If None, no calibration')
    parser.add_argument('--modify_alpha', type=float, default=1e-15)
    parser.add_argument('--modify_beta', type=float, default=1e-15)
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--save_statistics', action='store_true', help='save parameters of normal distribution when ipca is used')
    parser.add_argument('--save_holdout', action='store_true')
    parser.add_argument('--task_bdry', action='store_true', help='True if task bdry is known during training')
    parser.add_argument('--outlier_exposure', type=str, default='label', choices=['uniform', 'label'])
    parser.add_argument('--output_learning', type=int, default=None, help='number of samples to save to learn outputs')
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--folder', type=str, default=None, help='directory NAME. e.g. save under ./logs/NAME')
    parser.add_argument('--ff', type=float, default=1.)
    parser.add_argument('--dynamic', type=int, default=None, help='Set the max memory size. If set, use dynamic memory. Only works for PCAs. Use buffer_size for other methods')
    parser.add_argument('--compute_md', action='store_true', help='If true, compute mahalanobis distance of features')
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--resume_id', type=int, default=None, help='resume id. If provided, training begins when task_id == resume_id')
    parser.add_argument('--resume', type=str, default=None, help='resume path')
    parser.add_argument('--train_clf', action='store_true')
    parser.add_argument('--train_ebd', action='store_true')
    parser.add_argument('--obtain_val_outputs', action='store_true')
    parser.add_argument('--obtain_val_outputs_comp', action='store_true')
    parser.add_argument('--test_model_name', type=str, default=None, help='model_task_, model_task_clf_')
    parser.add_argument('--class_order', type=int, default=0, help='class split. Choices=[0, 1, 2]')
    parser.add_argument('--train_clf_save_name', type=str, default='model_task_clf')
    parser.add_argument('--model_copy', action='store_true')
    parser.add_argument('--train_clf_id', type=int, default=None)
    parser.add_argument('--task_inference', type=str, default=None, choices=['entropy'])

    # Network
    parser.add_argument('--in_dim', type=int, default=512, help='feature size')
    parser.add_argument('--out_dim', type=int, default=1)
    parser.add_argument('--freeze_head', action='store_true', help="If true, don't update classifier")

    # DataLoader
    parser.add_argument('--pin_memory', action='store_false')
    parser.add_argument('--num_workers', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--minibatch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=512)

    parser.add_argument('--load_best_args', action='store_true')
    
    parser.add_argument('--set_lr', type=float, default=None)
    parser.add_argument('--set_batch', type=int, default=None, help='batch size for train dataset')
    parser.add_argument('--set_epochs', type=int, default=None, help='Force n_epoch to be the set epoch even when best_args is used')

    # The followings are hyper-params for DER++
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--set_beta', type=float, default=None)
    parser.add_argument('--set_alpha', type=float, default=None)
    parser.add_argument('--set_minibatch', type=int, default=None, help='batch size for memory')
    parser.add_argument('--buffer_size', type=int, default=200)
    parser.add_argument('--sampling', action='store_true')

    # The followings are hyper-params for OWM
    parser.add_argument('--clipgrad', type=float, default=10)
    parser.add_argument('--owm_alpha', type=float, nargs='*', default=[0.1])

    # The followings are hyper-params for HAL
    parser.add_argument('--hal_lambda', type=float, default=0.2)
    parser.add_argument('--hal_beta', type=float, default=0.5)
    parser.add_argument('--hal_gamma', type=float, default=0.1)
    parser.add_argument('--steps_on_anchors', type=int, default=100)
    parser.add_argument('--finetuning_epochs', type=int, default=1)

    # The followings are hyper-parameters for knowledge distillation
    parser.add_argument('--distillation', action='store_true')
    parser.add_argument('--T', type=float, default=2)
    parser.add_argument('--distill_lambda', type=float, default=0.25)

    parser.add_argument('--pretrained', type=str, default='./moco_v2_800ep_pretrain.pth.tar')

    # For one class classification
    parser.add_argument('--n_hreg', type=float, default=2)
    parser.add_argument('--lamda', type=float, default=0.5)

    # For hierarchy model
    parser.add_argument('--hlr', type=float, default=0.5)

    # For vitadapter + OOD approaches
    parser.add_argument('--compute_auc', action='store_true')
    parser.add_argument('--calibration', action='store_true')
    parser.add_argument('--use_md', action='store_true', help='use MD value for CIL prediction')
    parser.add_argument('--noise', action='store_true', help='use MD-noise')
    parser.add_argument('--cal_lr', type=float, default=0.01)
    parser.add_argument('--cal_batch_size', type=int, default=8)
    parser.add_argument('--cal_epochs', type=int, default=5)
    parser.add_argument('--cal_size', type=int, default=20, help='number of samples saved per class for calibration')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value for sgd')
    parser.add_argument('--adapter_latent', type=int, default=64, help='adapter latent size')
    parser.add_argument('--softmax', action='store_true', help='use softmax for task output before cat for CIL')
    parser.add_argument('--lamb', type=float, default=0.9)

    # For HAT
    parser.add_argument('--smax', type=float, default=500)
    parser.add_argument('--lamb0', type=float, default=0.75)
    parser.add_argument('--lamb1', type=float, default=0.75)
    parser.add_argument('--thres_cosh', type=float, default=50)
    parser.add_argument('--thres_emb', type=float, default=6)

    # For PASS
    parser.add_argument('--kd_weight', type=float, default=10.0)
    parser.add_argument('--protoAug_weight', type=float, default=10.0)
    parser.add_argument('--pass_ensemble', action='store_true')

    # For DER
    parser.add_argument('--mem_size_mode', type=str, default="uniform_fixed_total_mem")
    parser.add_argument('--fixed_memory_per_cls', type=int, default=20)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--scheduler', type=str, default='multistep')
    parser.add_argument('--scheduling', type=float, nargs='*', default=[100, 120])
    parser.add_argument('--use_aux_cls', action='store_true')
    parser.add_argument('--aux_n_1', action='store_true')

    args = parser.parse_args()
    if args.dataset == 'mnist':
        args.total_cls = 10
    elif args.dataset == 'cifar10':
        args.total_cls = 10
    elif args.dataset == 'cifar100':
        args.total_cls = 100
    elif args.dataset == 'timgnet':
        args.total_cls = 200
    elif args.dataset == 'imagenet':
        args.total_cls = 1000

    if args.load_best_args:
        best = best_args[args.dataset][args.model]
        for k, v in best.items():
            setattr(args, k, v)

        if args.set_epochs is not None:
            args.n_epochs = args.set_epochs
        if args.set_beta is not None:
            args.beta = args.set_beta
        if args.set_alpha is not None:
            args.alpha = args.set_alpha
        if args.set_batch is not None:
            args.batch_size = args.set_batch
        if args.set_minibatch is not None:
            args.minibatch_size = args.set_minibatch
        if args.set_lr is not None:
            args.lr = args.set_lr

    return args
