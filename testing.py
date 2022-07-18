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

def test(task_list, args, train_data, test_data, model):
    # noise cannot be used without use_md
    if args.noise: assert args.use_md

    zeroshot = Zeroshot(args.model_clip, args)

    cil_tracker = Tracker(args)
    cal_cil_tracker = Tracker(args)
    til_tracker = Tracker(args)
    auc_softmax_tracker = AUCTracker(args)
    openworld_softmax_tracker = OWTracker(args)

    # cil_correct, til_correct are for cumulative accuracy throughout training
    cil_correct, til_correct, total = 0, 0, 0

    c_correct, c_total, p_correct, p_total = 0, 0, 0, 0
    cum_acc_list, total_loss_list, iter_list, total_iter = [], [], [], 0

    train_loaders, test_loaders, calibration_loaders = [], [], []

    args.mean, args.cov, args.cov_inv = {}, {}, {}
    args.mean_task, args.cov_noise, args.cov_inv_noise = {}, {}, {}

    # calibration weight w and bias b
    w, b = None, None

    param_copy = None

    combined_sigma = 0

    if args.task_type == 'concept': if_shift = []

    for task_id in range(args.load_task_id + 1):
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

        if hasattr(model, 'preprocess_task'):
            model.preprocess_task(names=train_data.task_list[task_id][0],
                                  labels=train_data.task_list[task_id][1],
                                  task_id=task_id,
                                  loader=train_loaders[-1])

        # Load model
        if args.test_model_name is None:
            test_model_name = 'model_task_'
        else:
            test_model_name = args.test_model_name
        if os.path.exists(args.load_dir + '/' + test_model_name + str(task_id)):
            filename = args.load_dir + '/' + test_model_name + str(task_id)
            args.logger.print("Load a trained model from:")
            args.logger.print(filename)
            state_dict = torch.load(filename)          
            model.net.load_state_dict(state_dict)

            if args.train_clf_id is not None:
                if task_id <= args.train_clf_id:
                    filename = args.load_dir + '/' + f'model_task_clf_epoch=10_{task_id}'
                else:
                    filename = args.load_dir + '/' + f'model_task_clf_epoch=10_{args.train_clf_id}'
                filename = torch.load(filename)

                for n, p in model.net.named_parameters():
                    if 'head' in n:
                        if n in filename.keys():
                            args.logger.print("changed head:", n)
                            print('before', torch.sum(p), torch.sum(filename[n]))
                            p.data = filename[n].data
                            print('after', torch.sum(p), torch.sum(filename[n]))

        else:
            raise NotImplementedError(args.load_dir + '/' + test_model_name + str(task_id), "Load dir incorrect")

        # Load statistics for MD
        if os.path.exists(args.load_dir + f'/cov_task_{task_id}.npy'):
            args.compute_md = True
            args.logger.print("*** Load Statistics for MD ***")
            cov = np.load(args.load_dir + f'/cov_task_{task_id}.npy')
            args.cov[task_id] = cov
            args.cov_inv[task_id] = np.linalg.inv(cov)
            if args.noise:
                mean = np.load(args.load_dir + f'/mean_task_{task_id}.npy')
                args.mean_task[task_id] = mean
                cov = np.load(args.load_dir + f'/cov_task_noise_{task_id}.npy')
                args.cov_noise[task_id] = cov
                args.cov_inv_noise[task_id] = np.linalg.inv(cov)
            for y in range(task_id * args.num_cls_per_task, (task_id + 1) * args.num_cls_per_task):
                mean = np.load(args.load_dir + f'/mean_label_{y}.npy')
                args.mean[y] = mean
        else:
            args.logger.print("*** No MD ***")

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
            args.logger.print(param_copy.sum(1))

        # Check/load if calibration is saved
        if os.path.exists(args.load_dir + f'/w_b_task_{task_id}'):
            w = torch.load(args.load_dir + f'/w_b_task_{task_id}')[0].to(args.device)
            b = torch.load(args.load_dir + f'/w_b_task_{task_id}')[1].to(args.device)

        if args.train_clf_id is not None:
            if task_id <= args.train_clf_id:
                continue
        for x, y, _, _, _ in test_loaders[-1]: # THIS NEEDS TO BE CHANGE (FROM TEST TO TRAINLOADER) WHEN SAVING FEATURES
            x, y = x.to(args.device), y.to(args.device)
            with torch.no_grad():
                if args.model_clip:
                    x = args.model_clip.encode_image(x).type(torch.FloatTensor).to(args.device)
                if args.zero_shot:
                    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in train_data.seen_names]).to(args.device)
                    zeroshot.evaluate(x, text_inputs, y)
            model.evaluate(x, y, task_id, w=w, b=b, total_learned_task_id=task_id, ensemble=args.pass_ensemble)
        metrics = model.acc()
        cil_tracker.update(metrics['cil_acc'], task_id, task_id)
        til_tracker.update(metrics['til_acc'], task_id, task_id)
        cal_cil_tracker.update(metrics['cal_cil_acc'], task_id, task_id)

        if args.compute_auc:
            in_scores = metrics['scores']
            if args.compute_md: in_scores_md = metrics['scores_md']
            auc_list, auc_list_md = [], []
            auc_total_in_list, auc_total_out_list, out_id_list = [metrics['scores_total']], [], []
            for task_out in range(args.n_tasks):
                if task_out != task_id:
                    if args.validation is None:
                        t_test = test_data.make_dataset(task_out)
                    else:
                        _, t_test = train_data.make_dataset(task_out)
                    ood_loader = make_loader(t_test, args, train='test')
                    for x, y, _, _, _ in ood_loader:
                        x, y = x.to(args.device), y.to(args.device)
                        with torch.no_grad():
                            model.evaluate(x, y, task_id=task_id, total_learned_task_id=task_id, ensemble=args.pass_ensemble)
                    metrics = model.acc()

                    if task_out <= task_id:
                        cil_tracker.update(metrics['cil_acc'], task_id, task_out)
                        til_tracker.update(metrics['til_acc'], task_id, task_out)

                    out_scores = metrics['scores']
                    auc = compute_auc(in_scores, out_scores)
                    auc_list.append(auc * 100)
                    args.logger.print("in/out: {}/{} | Softmax AUC: {:.2f}".format(task_id, task_out, auc_list[-1]), end=' ')
                    auc_softmax_tracker.update(auc_list[-1], task_id, task_out)

                    if args.compute_md:
                        out_scores_md = metrics['scores_md']
                        auc_md = compute_auc(in_scores_md, out_scores_md)
                        auc_list_md.append(auc_md * 100)
                        args.logger.print("| MD AUC: {:.2f}".format(auc_list_md[-1]))
                    else:
                        args.logger.print('')

                    if task_out <= task_id:
                        auc_total_in_list.append(metrics['scores_total'])
                    else:
                        auc_total_out_list.append(metrics['scores_total'])
                        out_id_list.append(task_out)

            args.logger.print("Average Softmax AUC: {:.2f}".format(np.array(auc_list).mean()))
            if args.compute_md:
                args.logger.print("Average MD AUC: {:.2f}".format(np.array(auc_list_md).mean()))

            for task_out, out_scores in zip(out_id_list, auc_total_out_list):
                auc = compute_auc(auc_total_in_list, out_scores)
                args.logger.print("total in/out: {}/{} | AUC: {:.2f}".format(task_id, task_out, auc * 100))
                openworld_softmax_tracker.update(auc * 100, task_id, task_out)
            if len(auc_total_in_list) > 0 and len(auc_total_out_list) > 0:
                auc = compute_auc(auc_total_in_list, auc_total_out_list)
                args.logger.print("total in | AUC: {:.2f}".format(auc * 100))

        # Report CIL, TIL, Cal_CIL. If compute_auc is True, they've already computed
        args.logger.print("######################")
        args.logger.print()
        if args.compute_auc:
            args.logger.print("Softmax AUC result")
            auc_softmax_tracker.print_result(task_id, type='acc')
            args.logger.print("Open World result")
            openworld_softmax_tracker.print_result(task_id, type='acc')
        args.logger.print("CIL result")
        cil_tracker.print_result(task_id, type='acc')
        cil_tracker.print_result(task_id, type='forget')
        args.logger.print("TIL result")
        til_tracker.print_result(task_id, type='acc')
        til_tracker.print_result(task_id, type='forget')
        args.logger.print()

        if task_id == 0 and args.calibration:
            cal_cil_tracker.mat = deepcopy(cil_tracker.mat)
        if w is not None:
            args.logger.print("Cal CIL result")
            cal_cil_tracker.print_result(task_id, type='acc')
            cal_cil_tracker.print_result(task_id, type='forget')

        torch.save(cil_tracker.mat, args.logger.dir() + '/cil_tracker_train_clf_equal_test')
        torch.save(til_tracker.mat, args.logger.dir() + '/til_tracker_train_clf_equal_test')
        torch.save(auc_softmax_tracker.mat, args.logger.dir() + '/auc_softmax_tracker_train_clf_equal_test')
        torch.save(openworld_softmax_tracker.mat, args.logger.dir() + '/openworld_softmax_tracker_train_clf_equal_test')

    return cil_tracker.mat, til_tracker.mat, cal_cil_tracker.mat
