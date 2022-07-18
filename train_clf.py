import timm
from numpy.linalg import svd
from torch.optim import SGD, Adam
from collections import Counter
from itertools import chain
from utils.utils import *
import torch
import clip
import matplotlib.pyplot as plt
from torch.autograd import Variable
from numpy.random import multivariate_normal
from tqdm import tqdm
from utils.my_ipca import MyIPCA as IPCA
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.utils.data import Subset
from collections import Counter

def train(task_list, args, train_data, test_data, model):
    # noise cannot be used without use_md
    if args.noise: assert args.use_md

    zeroshot = Zeroshot(args.model_clip, args)

    cil_tracker = Tracker(args)
    til_tracker = Tracker(args)
    cal_cil_tracker = Tracker(args)

    # cil_correct, til_correct are for cumulative accuracy throughout training
    cil_correct, til_correct, total = 0, 0, 0

    c_correct, c_total, p_correct, p_total = 0, 0, 0, 0
    cum_acc_list, total_loss_list, iter_list, total_iter = [], [], [], 0

    train_loaders, test_loaders, calibration_loaders = [], [], []

    args.mean, args.cov, args.cov_inv = {}, {}, {}
    args.mean_task, args.cov_noise, args.cov_inv_noise = {}, {}, {}

    param_copy = None

    combined_sigma = 0

    if args.task_type == 'concept': if_shift = []

    for task_id in range(len(task_list)):
        task_loss_list = []

        if args.validation is None:
            t_train = train_data.make_dataset(task_id)
            t_test = test_data.make_dataset(task_id)
        else:
            t_train, t_test = train_data.make_dataset(task_id)

        if args.calibration:
            assert args.cal_batch_size > 0
            assert args.cal_epochs > 0
            assert args.cal_size > 0
            t_train, t_cal = calibration_dataset(args, t_train)
            calibration_loaders.append(make_loader(t_cal, args, train='calibration'))

        train_loaders.append(make_loader(t_train, args, train='train'))
        test_loaders.append(make_loader(t_test, args, train='test'))
        if task_id > 0:
            if args.use_buffer:
                memory = torch.load(args.logger.dir() + f'/memory_{task_id - 1}')
                model.buffer_dataset.data = memory[0]
                model.buffer_dataset.targets = memory[1]
                model.buffer_dataset.transform = train_loaders[-1].dataset.transform

        if hasattr(model, 'preprocess_task'):
            model.preprocess_task(names=train_data.task_list[task_id][0],
                                  labels=train_data.task_list[task_id][1],
                                  task_id=task_id,
                                  loader=train_loaders[-1])
        state_dict = torch.load(args.logger.dir() + f'/model_task_{task_id}')
        model.net.load_state_dict(state_dict)

        # Load statistics for MD
        if os.path.exists(args.logger.dir() + f'/cov_task_{task_id}.npy'):
            args.compute_md = True
            args.logger.print("*** Load Statistics for MD ***")
            cov = np.load(args.logger.dir() + f'/cov_task_{task_id}.npy')
            args.cov[task_id] = cov
            args.cov_inv[task_id] = np.linalg.inv(cov)
            if args.noise:
                mean = np.load(args.logger.dir() + f'/mean_task_{task_id}.npy')
                args.mean_task[task_id] = mean
                cov = np.load(args.logger.dir() + f'/cov_task_noise_{task_id}.npy')
                args.cov_noise[task_id] = cov
                args.cov_inv_noise[task_id] = np.linalg.inv(cov)
            for y in range(task_id * args.num_cls_per_task, (task_id + 1) * args.num_cls_per_task):
                mean = np.load(args.logger.dir() + f'/mean_label_{y}.npy')
                args.mean[y] = mean
        else:
            args.logger.print("*** No MD ***")

        if args.distillation:
            raise NotImplementedError("model name not matching")

        if args.task_type == 'concept':
            if 'shifted' in train_data.current_labels:
                args.logger.print(train_data.current_labels)
                if_shift.append(True)
                init = int(train_data.current_labels.split('shifted: ')[-1].split(' -> ')[0])

                test_loaders[init].dataset.update()
                args.logger.print(len(test_loaders[init].dataset.targets))
            else:
                if_shift.append(False)

        if args.modify_previous_ood and task_id > 0:
            assert args.model == 'oe' or args.model == 'oe_fixed_minibatch'
            param_copy = model.net.fc.weight.detach()
            print(param_copy.sum(1))

        if args.use_buffer and task_id > 0:
            feature_list, label_list, output_list = [], [], []
            model_copy = None
            if args.model_copy:
                args.logger.print("Use model copy")
                model_copy = deepcopy(model.net)
            for p_task_id in range(task_id):
                mem = deepcopy(model.buffer_dataset)
                sample_per_cls = Counter(mem.targets)[0]
                args.logger.print("********************* samples per class:",
                                    sample_per_cls,
                                    "***********************")
                # model.buffer_dataset.data = train_loaders[0].dataset.data
                # model.buffer_dataset.targets = train_loaders[0].dataset.targets

                length = len(train_loaders[-1].dataset.targets)
                uniques = np.unique(train_loaders[-1].dataset.targets)
                idx = []
                for y_ in uniques:
                    idx.append(np.random.choice(np.where(train_loaders[-1].dataset.targets == y_)[0], size=sample_per_cls, replace=False))
                idx = np.concatenate(idx)

                if isinstance(train_loaders[-1].dataset, Subset):
                    for k in range(len(mem.data)):
                        mem.data[k] = 'data'.join([train_loaders[-1].dataset.dataset.samples[0][0].split('data')[0],
                                                mem.data[k].split('data')[1],])
                        mem.loader = t_test.dataset.loader

                    idx = train_loaders[-1].dataset.indices[idx]
                    for k in idx:
                        mem.data.append(train_loaders[-1].dataset.dataset.samples[k][0])
                        mem.targets.append(train_loaders[-1].dataset.dataset.samples[k][1])
                else:
                    mem.data = np.concatenate((mem.data, train_loaders[-1].dataset.data[idx]))
                    mem.targets = np.concatenate((mem.targets, train_loaders[-1].dataset.targets[idx]))

                loader = make_loader(mem, args, train='train')
                model.net.train()

                args.logger.print(Counter(loader.dataset.targets))

                model_ref = deepcopy(model.net.head[p_task_id])
                for epoch in range(args.n_epochs):
                    total_loss = 0
                    for x, y in loader:
                        x, y = x.to(args.device), y.to(args.device)
                        with torch.no_grad():
                            x, _ = model.net.forward_features(0, x, s=args.smax)
                        total_loss += model.train_clf(x, y, model_ref, p_task_id, model_copy)
                    args.logger.print("Classifier training. Task {}, Epoch {}/{}, Averag loss: {:.4f}".format(p_task_id,
                                                        epoch + 1, args.n_epochs, total_loss / len(loader)))
                    print(model.net.head[0].weight.sum())


                args.logger.print("task id:", p_task_id)
                # Evaluate current task
                model.reset_eval()
                for x, y, _, _, _ in test_loaders[p_task_id]:
                    x, y = x.to(args.device), y.to(args.device)
                    with torch.no_grad():
                        if args.model_clip:
                            x = args.model_clip.encode_image(x).type(torch.FloatTensor).to(args.device)
                        if args.zero_shot:
                            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_data.seen_names]).to(args.device)
                            zeroshot.evaluate(x, text_inputs, y)
                    model.evaluate(x, y, p_task_id, report_cil=True, total_learned_task_id=p_task_id, ensemble=args.pass_ensemble)
                metrics = model.acc()
                args.logger.print("Task {}, Epoch {}/{}, Total Loss: {:.4f}, CIL Acc: {:.2f}, TIL Acc: {:.2f}".format(p_task_id,
                                    epoch + 1, args.n_epochs, np.mean(task_loss_list),
                                    metrics['cil_acc'], metrics['til_acc']))

        # End task
        if hasattr(model, 'end_task'):
            if args.calibration:
                model.end_task(calibration_loaders, test_loaders, train_loader=train_loaders[-1])
            else:
                model.end_task(task_id + 1, train_loader=train_loaders[-1])

        # Save
        torch.save(model.net.state_dict(),
                    args.logger.dir() + f'{args.train_clf_save_name}_{task_id}')
        if args.calibration:
            if model.w is not None:
                torch.save(model.w.data,
                            args.logger.dir() + f'calibration_w_task_{task_id}')
                torch.save(model.b.data,
                            args.logger.dir() + f'calibration_b_task_{task_id}')

        # Save statistics e.g. mean, cov, cov_inv
        if args.save_statistics:
            np.save(args.logger.dir() + 'statistics', model.statistics)

        args.logger.print("######################")
        true_lab, pred_lab = [], []
        for p_task_id, loader in enumerate(test_loaders):
            model.reset_eval()
            for x, y, _, _, _ in loader:
                x, y = x.to(args.device), y.to(args.device)
                with torch.no_grad():
                    if args.model_clip:
                        x = args.model_clip.encode_image(x).type(torch.FloatTensor).to(args.device)
                model.evaluate(x, y, task_id=p_task_id, report_cil=True, total_learned_task_id=task_id, ensemble=args.pass_ensemble)
            if args.save_output:
                np.save(args.logger.dir() + 'output_learned_{}_task_{}'.format(task_id, p_task_id),
                                                        np.concatenate(model.output_list))
                np.save(args.logger.dir() + 'label_learned_{}_task_{}'.format(task_id, p_task_id),
                                                        np.concatenate(model.label_list))

            metrics = model.acc()
            cil_tracker.update(metrics['cil_acc'], task_id, p_task_id)
            til_tracker.update(metrics['til_acc'], task_id, p_task_id)

            if args.tsne:
                tsne(np.concatenate(model.output_list),
                     np.concatenate(model.label_list),
                     logger=args.logger)
            if args.confusion:
                true_lab_ = np.concatenate(model.true_lab)
                pred_lab_ = np.concatenate(model.pred_lab)

                plot_confusion(true_lab_, pred_lab_, model.seen_names, task_id,
                                p_task_id, logger=args.logger,
                                num_cls_per_task=args.num_cls_per_task)

                true_lab.append(true_lab_)
                pred_lab.append(pred_lab_)

            if args.confusion and p_task_id == len(test_loaders) - 1:
                true_lab_ = np.concatenate(true_lab)
                pred_lab_ = np.concatenate(pred_lab)
                plot_confusion(true_lab_, pred_lab_, model.seen_names,
                                name='confusion mat task {}'.format(p_task_id),
                                logger=args.logger, num_cls_per_task=args.num_cls_per_task)

        args.logger.print()
        args.logger.print("CIL result")
        cil_tracker.print_result(task_id, type='acc')
        cil_tracker.print_result(task_id, type='forget')
        args.logger.print("TIL result")
        til_tracker.print_result(task_id, type='acc')
        til_tracker.print_result(task_id, type='forget')
        args.logger.print()

        if task_id == 0 and args.calibration:
            model.cil_acc_mat_test = deepcopy(cil_tracker.mat)

        torch.save(cil_tracker.mat, args.logger.dir() + '/cil_tracker_train_clf_equal')
        torch.save(til_tracker.mat, args.logger.dir() + '/til_tracker_train_clf_equal')

    plt.plot(cum_acc_list)
    xticks = [l[0] for l in iter_list]
    xticks.append(iter_list[-1][-2])
    plt.xticks(xticks)
    plt.xlabel('Training Time')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Cumulative Accuracy over Training Time')
    plt.savefig(args.logger.dir() + 'cumulative_acc.png')
    plt.close()