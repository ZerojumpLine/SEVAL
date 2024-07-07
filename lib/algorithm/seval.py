import json
import os
import torch.nn as nn
import numpy as np

import time

import torch
from lib.models import EMAModel
from yacs.config import CfgNode
from lib.models.losses import Logitdiff, OptimizePi, OptimizeTH
from lib.models.feature_queue import FeatureQueue
from lib.utils import Meters, get_last_n_median
from lib.dataset.utils import get_data_config
from .base_ssl_algorithm import SemiSupervised
from lib.models.losses import Accuracy

import torch.nn.functional as F
from .base_ssl_algorithm import cal_acc

class SEVAL(SemiSupervised):

    def __init__(self, cfg: CfgNode) -> None:
        super().__init__(cfg)

        self.record = cfg.PERIODS.RECORD
        self.alpha = torch.ones(self.num_classes)
        self.alpha_current = torch.ones(self.num_classes)
        self.pi = torch.ones(self.num_classes)
        self.pi_current = torch.ones(self.num_classes)
        if self.device is not None:
            self.alpha = self.alpha.to(self.device)
            self.alpha_current = self.alpha.to(self.device)
            self.pi = self.pi.to(self.device)
            self.pi_current = self.pi_current.to(self.device)

        self.conf_thres = [cfg.ALGORITHM.CONFIDENCE_THRESHOLD] * self.num_classes
        
        self.opt_func = cfg.ALGORITHM.SEVAL.OPT_FUNC
        self.threshold_start = cfg.ALGORITHM.SEVAL.TH_START
        self.threshold_end = cfg.ALGORITHM.SEVAL.TH_END
        
        self.pi_ema_decay = cfg.ALGORITHM.SEVAL.PI_EMA_DECAY
        self.th_ema_decay = cfg.ALGORITHM.SEVAL.TH_EMA_DECAY

        self.pi_bounds = [cfg.ALGORITHM.SEVAL.ALPHA_LOW, cfg.ALGORITHM.SEVAL.ALPHA_UP, 
                          cfg.ALGORITHM.SEVAL.PI_LOW, cfg.ALGORITHM.SEVAL.PI_UP]

        self.opt_pi = cfg.ALGORITHM.SEVAL.OPT_PI
        self.opt_th = cfg.ALGORITHM.SEVAL.OPT_TH
        self.pi_warmup_ratio = cfg.ALGORITHM.SEVAL.PI_WARMUP_RATIO
        self.pi_optimizer = cfg.ALGORITHM.SEVAL.PI_OPTIMITZER

        if cfg.ALGORITHM.SEVAL.PI_CLS != -1:
            self.pi_cls = cfg.ALGORITHM.SEVAL.PI_CLS
        else:
            self.pi_cls = self.num_classes
        
        if cfg.ALGORITHM.SEVAL.TH_CLS != -1:
            self.th_cls = cfg.ALGORITHM.SEVAL.TH_CLS
        else:
            self.th_cls = self.num_classes

        # set up the estim paths if the parameters are learned separately
        self.load_param = cfg.ALGORITHM.SEVAL.LOAD_PARAM
        self.abc = cfg.ALGORITHM.ABC.APPLY
        self.daso = cfg.ALGORITHM.DASO.APPLY
        self.estim_param = cfg.ALGORITHM.SEVAL.ESTIM.APPLY
        if self.load_param or self.estim_param:
            data_cfg = get_data_config(cfg)
            self.estim_path = cfg.ALGORITHM.SEVAL.EST
            num_l_head = data_cfg.NUM_LABELED_HEAD
            num_ul_head = data_cfg.NUM_UNLABELED_HEAD
            imb_factor_l = data_cfg.IMB_FACTOR_L
            imb_factor_ul = data_cfg.IMB_FACTOR_UL
            reverse_ul = cfg.DATASET.REVERSE_UL_DISTRIBUTION
            est_name = f"{cfg.DATASET.NAME}_l_{num_l_head}_{imb_factor_l}_" + \
                f"ul_{num_ul_head}_{imb_factor_ul}_seed_{cfg.SEED}_"
            if reverse_ul:
                est_name += "rev_"
            if self.abc:
                est_name += "abc_"
            if cfg.DATASET.NAME == "aves":
                if cfg.DATASET.AVES.UL_SOURCE == "in":
                    est_name += "in_"
                else:
                    est_name += "all_"
            self.est_name_alpha = est_name + "estim_alpha.npy"
            self.est_name_pi = est_name + "estim_pi.npy"
            self.est_name_thres = est_name + "estim_thres.npy"
            self.alpha_iters = []
            self.pi_iters = []
            self.conf_thres_iters = []
        
        if self.abc:
            class_count = self.get_label_dist(device=self.device)

            # we believe the last element to be the most tail class.
            self.bal_param = class_count[-1] / class_count  # bernoulli parameter

            self.alpha_abc = torch.ones(self.num_classes)
            self.alpha_abc_current = torch.ones(self.num_classes)
            self.pi_abc = torch.ones(self.num_classes)
            self.pi_abc_current = torch.ones(self.num_classes)
            if self.device is not None:
                self.alpha_abc = self.alpha_abc.to(self.device)
                self.alpha_abc_current = self.alpha_abc_current.to(self.device)
                self.pi_abc = self.pi_abc.to(self.device)
                self.pi_abc_current = self.pi_abc_current.to(self.device)
            self.conf_thres_abc = [cfg.ALGORITHM.CONFIDENCE_THRESHOLD] * self.num_classes

            if self.load_param or self.estim_param:
                self.est_name_alpha_abc = est_name + "abc_estim_pi.npy"
                self.est_name_pi_abc = est_name + "abc_estim_pi.npy"
                self.est_name_thres_abc = est_name + "abc_estim_thres.npy"
                self.alpha_iters_abc = []
                self.pi_iters_abc = []
                self.conf_thres_iters_abc = []
        
        if self.daso:
            self.pretrain_steps = cfg.ALGORITHM.DASO.PRETRAIN_STEPS
            self.pretraining = True
            self.similarity_fn = nn.CosineSimilarity(dim=2)        
            self.queue = FeatureQueue(cfg, classwise_max_size=None, bal_queue=True)
            self.psa_loss_weight = cfg.ALGORITHM.DASO.PSA_LOSS_WEIGHT
            self.T_proto = cfg.ALGORITHM.DASO.PROTO_TEMP

    def train(self):
        self.logger.info(f"Starting training from iteration {self.start_iter}")
        self.model.train()

        for self.iter in range(self.start_iter, self.max_iter):
            if (
                self.cfg.MODEL.LOSS.WITH_LABELED_COST_SENSITIVE
                and (self.iter + 1) >= self.cfg.MODEL.LOSS.WARMUP_ITERS and not self.is_warmed
            ):
                # warmup, LDAM-DRW (deferred reweight)
                self.is_warmed = True
                self.l_loss = self.build_labeled_loss(self.cfg, warmed_up=True)

            # one step of forward path and backprop
            self.run_step()

            # increase the meter's iteration
            self.meters.step()

            # eval period
            if ((self.iter + 1) % self.cfg.PERIODS.EVAL == 0):

                self.dist_logger.write()

                # Here I want to track the status of validation logits.
                self.track_logits(self.model)

                if not self.load_param:
                    
                    # Here I optimize pi
                    if self.opt_pi:
                        self.optimize_pi(self.ema_model)

                    if self.opt_th:
                        self.optimize_threshold(self.ema_model)

                    if self.estim_param and (self.opt_th or self.opt_pi):
                        # learn the parameters and also save
                        # self.pi, self.threshold
                        self.alpha_iters.append(np.array(self.alpha.cpu()))
                        self.pi_iters.append(np.array(self.pi.cpu()))
                        self.conf_thres_iters.append(self.conf_thres.copy())
                        np.save(os.path.join(self.estim_path, self.est_name_alpha), np.array(self.alpha_iters)) # iters x Class
                        np.save(os.path.join(self.estim_path, self.est_name_pi), np.array(self.pi_iters)) # iters x Class
                        np.save(os.path.join(self.estim_path, self.est_name_thres), np.array(self.conf_thres_iters)) # iters x Class
                        if self.abc:
                            self.alpha_iters_abc.append(np.array(self.alpha_abc.cpu()))
                            self.pi_iters_abc.append(np.array(self.pi_abc.cpu()))
                            self.conf_thres_iters_abc.append(self.conf_thres_abc.copy())
                            np.save(os.path.join(self.estim_path, self.alpha_iters_abc), np.array(self.alpha_iters_abc)) # iters x Class
                            np.save(os.path.join(self.estim_path, self.est_name_pi_abc), np.array(self.pi_iters_abc)) # iters x Class
                            np.save(os.path.join(self.estim_path, self.est_name_thres_abc), np.array(self.conf_thres_iters_abc)) # iters x Class

                else:
                    # load the paramters.
                    evalnum = int((self.iter + 1) / self.cfg.PERIODS.EVAL - 1)
                    est_alpha_iters = np.load(os.path.join(self.estim_path, self.est_name_alpha)) # iters x Class
                    est_pi_iters = np.load(os.path.join(self.estim_path, self.est_name_pi)) # iters x Class
                    est_thres_iters = np.load(os.path.join(self.estim_path, self.est_name_thres)) # iters x Class
                    est_pi = torch.from_numpy(est_pi_iters[evalnum, :]).float()
                    est_alpha = torch.from_numpy(est_alpha_iters[evalnum, :]).float()
                    if self.device is not None:
                        est_pi = est_pi.to(self.device)
                        est_alpha = est_alpha.to(self.device)
                    # calculate the original estimated pi...a litter crumpy, maybe I should save the original next version?
                    self.alpha_current = (est_alpha - self.pi_ema_decay * self.alpha) / (1 - self.pi_ema_decay)
                    self.alpha = est_alpha
                    self.pi_current = (est_pi - self.pi_ema_decay * self.pi) / (1 - self.pi_ema_decay)
                    self.pi = est_pi
                    est_thres = est_thres_iters[evalnum, :]
                    result_thres = {}
                    result_alpha = {}
                    result_pi = {}
                    for cnt in range(self.num_classes):
                        self.conf_thres[cnt] = est_thres[cnt]
                        result_thres['threshold_' + str(cnt)] = self.conf_thres[cnt]
                        result_alpha['alpha' + str(cnt)] = np.array(self.alpha[cnt].cpu()).tolist()
                        result_pi['pi' + str(cnt)] = np.array(self.pi[cnt].cpu()).tolist()
                    self.meters.put_scalars(result_thres, show_avg=False, prefix="seval_threshold")
                    self.meters.put_scalars(result_alpha, show_avg=False, prefix="seval_refine")
                    self.meters.put_scalars(result_pi, show_avg=False, prefix="seval_refine")

                    if self.abc:
                        est_alpha_iters_abc = np.load(os.path.join(self.estim_path, self.est_name_alpha_abc)) # iters x Class
                        est_pi_iters_abc = np.load(os.path.join(self.estim_path, self.est_name_pi_abc)) # iters x Class
                        est_thres_iters_abc = np.load(os.path.join(self.estim_path, self.est_name_thres_abc)) # iters x Class
                        est_alpha_abc = torch.from_numpy(est_alpha_iters_abc[evalnum, :]).float()
                        est_pi_abc = torch.from_numpy(est_pi_iters_abc[evalnum, :]).float()
                        if self.device is not None:
                            est_pi_abc = est_pi_abc.to(self.device)
                            est_alpha_abc = est_alpha_abc.to(self.device)
                        self.alpha_abc_current = (est_alpha_abc - self.pi_ema_decay * self.alpha_abc) / (1 - self.pi_ema_decay)
                        self.alpha_abc = est_alpha_abc
                        self.pi_abc_current = (est_pi_abc - self.pi_ema_decay * self.pi_abc) / (1 - self.pi_ema_decay)
                        self.pi_abc = est_pi_abc
                        est_thres_abc = est_thres_iters_abc[evalnum, :]
                        for cnt in range(self.num_classes):
                            self.conf_thres_abc[cnt] = est_thres_abc[cnt]
                
                self.evaluate(self.model)

                if self.record:
                    # calculate correctness and gain.
                    acc_test = self.eval_history["test/top1"][-1]
                    ul_preds, ul_labels, _ = self.eval_ul_dataset()
                    # ul_preds: N x D
                    # ul_labels: N

                    # pre-adjustment
                    acc_pre, m_acc_pre = cal_acc(ul_preds, ul_labels)

                    # post-adjustment

                    ul_preds = self.alpha.cpu() * (ul_preds - self.pi.cpu())    
                    acc_post, m_acc_post = cal_acc(ul_preds, ul_labels)
                
                    conf, pred_class = torch.max(ul_preds.detach().softmax(dim=1), dim=1)
                    ul_preds = np.array(ul_preds)
                    ul_labels = np.array(ul_labels)
                    class_weight = 1 / np.array(self.p_data.cpu())

                    conf = np.array(conf)
                    threshold_pos = []
                    for kbatch in range(len(conf)):
                        if conf[kbatch] > self.conf_thres[pred_class[kbatch]]:
                            threshold_pos.append(kbatch)
                    threshold_pos = np.array(threshold_pos)
                    
                    true_class = ul_labels[threshold_pos]
                    target_class = np.argmax(ul_preds, axis = 1)[threshold_pos]

                    acc_numerator = 0
                    correct_y1_pos = np.where(true_class == target_class)[0]
                    for cor_pos in correct_y1_pos:
                        acc_numerator = acc_numerator + class_weight[true_class[cor_pos]]
                    acc_denominator_1 = 1e-6
                    for all_pos in range(len(ul_labels)):
                        acc_denominator_1 = acc_denominator_1 + class_weight[ul_labels[all_pos]]
                    acc_denominator_2 = 1e-6
                    for all_pos in range(len(threshold_pos)):
                        acc_denominator_2 = acc_denominator_2 + class_weight[true_class[all_pos]]

                    correctness = acc_numerator * acc_numerator / acc_denominator_1 / acc_denominator_2

                    self.correctness_iters.append(correctness)
                    self.gain_iters.append((m_acc_post-m_acc_pre)*100)
                    self.gain_iters.append((acc_post-acc_pre)*100)
                    self.acc_iters.append(acc_test)

                    print(f"pre adjustment accuracy: {acc_pre}")
                    print(f"post adjustment accuracy: {acc_post}")
                    print(f"pre adjustment balanced accuracy: {m_acc_pre}")
                    print(f"post adjustment balanced accuracy: {m_acc_post}")

                    np.savetxt(self.coorectness_savename, np.array(self.correctness_iters), delimiter = ',')
                    np.savetxt(self.gain_savename, np.array(self.gain_iters), delimiter = ',')
                    np.savetxt(self.acc_savename, np.array(self.acc_iters), delimiter = ',')

            # periodically save checkpoints
            if (
                self.cfg.PERIODS.CHECKPOINT > 0
                and (self.iter + 1) % self.cfg.PERIODS.CHECKPOINT == 0
            ):
                save_ema_model = self.with_ul
                if self.cfg.ALGORITHM.NAME == "DARP_ESTIM":
                    save_ema_model = False
                self.save_checkpoint(save_ema_model=save_ema_model)

            # print logs
            if (((self.iter + 1) % self.cfg.PERIODS.LOG == 0 or (self.iter + 1) == self.max_iter)):
                assert self.cfg.PERIODS.EVAL == self.cfg.PERIODS.LOG
                for writer in self.writers:
                    writer.write(self.meters)
                self.meters.reset()

            # start new generation after evaluation!
            if (self.iter + 1) % self.gen_period_steps == 0:
                crest_names = ["ReMixMatchCReST", "FixMatchCReST", "SEVALCReST"]
                with_crest = self.cfg.ALGORITHM.NAME in crest_names
                # new generation except for the last iteration
                if with_crest and (self.iter + 1) < self.max_iter:
                    self.new_generation()
        print()
        print()
        print()

        prefixes = ["valid/top1", "test/top1"]
        self.logger.info("Median 20 Results:")
        self.logger.info(
            ", ".join(
                f"{k}_median (20): {get_last_n_median(v, n=20):.2f}"
                for k, v in self.eval_history.items() if k in prefixes
            )
        )
        print()
        prefixes = ["valid/top1_la", "test/top1_la"]
        self.logger.info("Median 20 Results:")
        self.logger.info(
            ", ".join(
                f"Logit adjusted {k}_median (20): {get_last_n_median(v, n=20):.2f}"
                for k, v in self.eval_history.items() if k in prefixes
            )
        )
        print()

        # final checkpoint
        self.save_checkpoint(save_ema_model=self.with_ul)

        # test top1 and median print
        print()
        save_path = self.cfg.OUTPUT_DIR
        with open(os.path.join(save_path, "results.json"), "w") as f:
            eval_history = {k: v for k, v in self.eval_history.items()}
            f.write(json.dumps(eval_history, indent=4, sort_keys=True))
        self.logger.info(f"final results (results.json) saved on: {save_path}.")

        for writer in self.writers:
            writer.close()

    def track_logits(self, model):
        # local metric and meters
        logitdiff = Logitdiff(self.num_classes)

        model.eval()

        with torch.no_grad():
            for i, (images, target, _) in enumerate(self.valid_loader):
                if torch.cuda.is_available():
                    images = images.to(self.device)
                    target = target.to(self.device).long()

                outputs = model(images, is_train=False)  # logits
                # compute metrics using original logits
                logitdiff(outputs, target)

        self.meters.put_scalars(logitdiff.classwise, show_avg=False, prefix="valid_logits")
        self.meters.put_scalars(logitdiff.classwise_prob, show_avg=False, prefix="valid_meanprob")

        model.train()
    
    def optimize_threshold(self, model):
        # local metric and meters
        if self.estim_param:
            # note that in this case, the validation data is imbalanced
            p_data = np.array(self.p_data.cpu())
        else:
            p_data = np.ones(self.num_classes)

        current_thres = self.threshold_start + (self.threshold_end - self.threshold_start) * self.iter / self.max_iter

        optimizeth = OptimizeTH(self.num_classes, self.th_cls, current_thres, p_data)

        if self.abc:
            optimizeth_abc = OptimizeTH(self.num_classes, self.th_cls, current_thres, p_data)

        model.eval()

        with torch.no_grad():
            for i, (images, target, _) in enumerate(self.valid_loader):
                if torch.cuda.is_available():
                    images = images.to(self.device)
                    target = target.to(self.device).long()

                if self.abc:
                    outputs = model(images, is_train=True)
                    outputs_abc = model(images, is_train=False)
                    outputs_seval_abc = self.alpha_abc * (outputs_abc - self.pi_abc.log())
                    optimizeth_abc(outputs_seval_abc, target)
                else:
                    outputs = model(images, is_train=False)  # logits
                    # compute metrics using original logits
                outputs_seval = self.alpha * (outputs - self.pi.log())

                optimizeth(outputs_seval, target)
        
        optimze_res, effective_sample, correct_rate, ul_samples, rare_class = optimizeth.optimize()
        
        result_seval = {}
        for cnt, th_cls, ul_sample in zip(range(len(optimze_res)), optimze_res, ul_samples):
            self.conf_thres[cnt] = self.th_ema_decay * self.conf_thres[cnt] + (1 - self.th_ema_decay) * th_cls
            result_seval['threshold_' + str(cnt)] = self.conf_thres[cnt]
            result_seval['ul_samples_' + str(cnt)] = ul_sample
        result_seval['effective_sample'] = effective_sample
        result_seval['correct_rate'] = correct_rate
        self.meters.put_scalars(result_seval, show_avg=False, prefix="seval_threshold")

        if len(rare_class) > 0:
            print(f'classes {rare_class} are under representated')
            for r in rare_class:
                self.pi[r] = torch.min(self.pi)
                self.pi_current[r] = torch.min(self.pi_current)

        if self.abc:
            optimze_res_abc, _, _, _, _ = optimizeth_abc.optimize()

            for cnt, th_cls in zip(range(len(optimze_res_abc)), optimze_res):
                self.conf_thres_abc[cnt] = self.th_ema_decay * self.conf_thres_abc[cnt] + (1 - self.th_ema_decay) * th_cls

        model.train()
    
    def optimize_pi(self, model):
        # local metric and meters
        
        # warm up for label refinement.
        result_seval = {}

        if self.iter > int(self.max_iter * self.pi_warmup_ratio):
            optimizepi = OptimizePi(self.num_classes, self.pi_cls)
            if self.abc:
                optimizepi_abc = OptimizePi(self.num_classes, self.pi_cls)

            model.eval()

            with torch.no_grad():
                for i, (images, target, _) in enumerate(self.valid_loader):
                    if torch.cuda.is_available():
                        images = images.to(self.device)
                        target = target.to(self.device).long()
                    
                    if self.abc:
                        outputs = model(images, is_train=True)
                        outputs_abc = model(images, is_train=False)
                        optimizepi_abc(outputs_abc, target)
                    else:
                        outputs = model(images, is_train=False)  # logits
                    # compute metrics using original logits
                    optimizepi(outputs, target)
            
            optimze_res, acc_pre, acc_post = optimizepi.optimize(metric=self.opt_func, bounds=self.pi_bounds, optimizer=self.pi_optimizer)

            alpha = torch.from_numpy(optimze_res[:int(len(optimze_res)/2)]).float()
            pi = torch.from_numpy(optimze_res[int(len(optimze_res)/2):]).float()

            if self.device is not None:
                alpha = alpha.to(self.device)
                pi = pi.to(self.device)

            self.alpha = self.pi_ema_decay * self.alpha + (1 - self.pi_ema_decay) * alpha
            self.alpha_current = alpha
            self.pi = self.pi_ema_decay * self.pi + (1 - self.pi_ema_decay) * pi
            self.pi_current = pi

            if self.abc:
                optimze_res_abc, _, _ = optimizepi_abc.optimize(metric=self.opt_func, bounds=self.pi_bounds, optimizer=self.pi_optimizer)
                alpha_abc = torch.from_numpy(optimze_res_abc[:int(len(optimze_res)/2)]).float()
                pi_abc = torch.from_numpy(optimze_res_abc[int(len(optimze_res)/2):]).float()

                if self.device is not None:
                    alpha_abc = alpha_abc.to(self.device)
                    pi_abc = pi_abc.to(self.device)

                self.alpha_abc = self.pi_ema_decay * self.alpha_abc + (1 - self.pi_ema_decay) * alpha_abc
                self.alpha_abc_current = alpha_abc
                self.pi_abc = self.pi_ema_decay * self.pi_abc + (1 - self.pi_ema_decay) * pi_abc
                self.pi_abc_current = pi_abc

            result_seval['acc_pre'] = np.mean(acc_pre)
            result_seval['acc_post'] = np.mean(acc_post)
        
        for cnt in range(self.num_classes):
            result_seval['alpha' + str(cnt)] = np.array(self.alpha[cnt].cpu()).tolist()
            result_seval['pi' + str(cnt)] = np.array(self.pi[cnt].cpu()).tolist()
            result_seval['acc_gain' + str(cnt)] = acc_post[cnt] - acc_pre[cnt]
            
        self.meters.put_scalars(result_seval, show_avg=False, prefix="seval_refine")

        model.train()

    def eval_loop(self, model, data_loader, *, prefix: str = "valid") -> float:
        # local metric and meters
        accuracy = Accuracy(self.num_classes)
        accuracy_org = Accuracy(self.num_classes)
        accuracy_la = Accuracy(self.num_classes)
        accuracy_seval = Accuracy(self.num_classes)
        meters = Meters()

        model.eval()
        log_classwise = self.cfg.MISC.LOG_CLASSWISE
        with torch.no_grad():
            for i, (images, target, _) in enumerate(data_loader):
                metrics = {}
                if torch.cuda.is_available():
                    images = images.to(self.device)
                    target = target.to(self.device).long()

                outputs = model(images, is_train=False)  # logits
                batch_size = images.size(0)

                # compute metrics using original logits
                loss = F.cross_entropy(outputs, target, reduction="none").mean()
                top1, top5 = accuracy(outputs, target, log_classwise=log_classwise)
                _, _ = accuracy_org(outputs, target, log_classwise=log_classwise)
                metrics.update({"cost": loss.item(), "top1": top1, "top5": top5})

                # adjust logits in test-time and compute metrics again
                outputs_la = outputs - self.p_data.view(1, -1).log()
                loss_la = F.cross_entropy(outputs_la, target, reduction="none").mean()
                top1_la, top5_la = accuracy(
                    outputs_la, target, log_classwise=log_classwise, prefix="logit_adjusted"
                )
                _, _ = accuracy_la(
                    outputs_la, target, log_classwise=log_classwise, prefix="logit_adjusted"
                )
                metrics.update({"cost_la": loss_la.item(), "top1_la": top1_la, "top5_la": top5_la})
                meters.put_scalars(metrics, n=batch_size)

                # SEVAL refinement - current PI
                if self.abc:
                    outputs_seval_current = self.alpha_abc_current * (outputs - self.pi_abc_current.log())
                else:
                    outputs_seval_current = self.alpha_current * (outputs - self.pi_current.log())
                loss_seval_current = F.cross_entropy(outputs_seval_current, target, reduction="none").mean()
                top1_seval_current, top5_seval_current = accuracy(
                    outputs_seval_current, target, log_classwise=log_classwise, prefix="seval_current"
                )
                _, _ = accuracy_seval(
                    outputs_seval_current, target, log_classwise=log_classwise, prefix="seval_current"
                )
                metrics.update({"cost_seval_current": loss_seval_current.item(), 
                                "top1_seval_current": top1_seval_current, 
                                "top5_seval_current": top5_seval_current})
                meters.put_scalars(metrics, n=batch_size)

                # SEVAL refinement - moving averaged PI
                if self.abc:
                    outputs_seval = self.alpha_abc * (outputs - self.pi_abc.log())
                else:
                    outputs_seval = self.alpha * (outputs - self.pi.log())
                loss_seval = F.cross_entropy(outputs_seval, target, reduction="none").mean()
                top1_seval, top5_seval = accuracy(
                    outputs_seval, target, log_classwise=log_classwise, prefix="seval_ema"
                )
                metrics.update({"cost_seval": loss_seval.item(), "top1_seval": top1_seval, "top5_seval": top5_seval})
                meters.put_scalars(metrics, n=batch_size)



        # log classwise accuracy
        if log_classwise:
            self.meters.put_scalars(
                accuracy.classwise, show_avg=False, prefix=prefix + "_classwise"
            )
            # write the averaged recall (accuracy)
            recall_cls = list(accuracy.classwise.values())
            mean_recall = sum(recall_cls) / len(recall_cls)
            result_test = {}
            result_test['avg_recall'] = mean_recall
            self.meters.put_scalars(result_test, show_avg=False, prefix="test")
            # write the averaged recall (accuracy)
            recall_cls = list(accuracy_la.classwise.values())
            mean_recall = sum(recall_cls) / len(recall_cls)
            result_test = {}
            result_test['avg_recall_la'] = mean_recall
            self.meters.put_scalars(result_test, show_avg=False, prefix="test")
            # write the averaged recall (accuracy)
            recall_cls = list(accuracy_seval.classwise.values())
            mean_recall = sum(recall_cls) / len(recall_cls)
            result_test = {}
            result_test['avg_recall_seval'] = mean_recall
            self.meters.put_scalars(result_test, show_avg=False, prefix="test")


        # aggregate the metrics and log
        results = meters.get_latest_scalars_with_avg()
        self.eval_history[prefix + "/top1"].append(results["top1"])
        self.eval_history[prefix + "/top1_la"].append(results["top1_la"])

        model.train()

        return results

    def run_step(self) -> None:
        loss_dict = {}
        metrics_dict = {}

        # measure data time
        start = time.perf_counter()
        l_images, labels, _ = next(self._l_iter)

        (ul_weak, ul_strong), UL_LABELS, ul_indices = next(self._ul_iter)
        data_time = time.perf_counter() - start

        # load images and labels onto gpu
        if torch.cuda.is_available():
            l_images = l_images.to(self.device)
            labels = labels.to(self.device).long()
            ul_weak = ul_weak.to(self.device)
            ul_strong = ul_strong.to(self.device)
            UL_LABELS = UL_LABELS.to(self.device)

        num_labels = labels.size(0)

        if self.abc:

            # feature extraction
            input_concat = torch.cat([l_images, ul_weak, ul_strong], 0)
            feats_concat = self.model(input_concat, return_features=True)

            # logits for ABC
            logits_concat_abc = self.model.abc_classify(feats_concat)
            l_logits_abc = logits_concat_abc[:num_labels]

            l_mask_abc = torch.bernoulli(self.bal_param[labels].detach()).float()
            cls_loss_abc = self.l_loss(l_logits_abc, labels, weight=l_mask_abc)

            # unlabeled data part
            logits_weak_abc, logits_strong_abc = logits_concat_abc[num_labels:].chunk(2)
            logits_weak_abc = self.alpha_abc * (logits_weak_abc - self.pi_abc.log())
            p_abc = logits_weak_abc.detach().softmax(dim=1)  # soft pseudo labels
            with torch.no_grad():
                conf_abc, pred_class_abc = torch.max(p_abc, dim=1)
            
            loss_weight_abc = conf_abc
            for kbatch in range(len(conf_abc)):
                loss_weight_abc[kbatch] = conf_abc[kbatch].ge(self.conf_thres_abc[pred_class_abc[kbatch].int()]).float()

            # mask generation
            epoch_iters = self.cfg.PERIODS.EVAL
            epoch_nums = int(self.cfg.SOLVER.MAX_ITER / epoch_iters)
            current_epoch = int(self.iter / epoch_iters)  # 0~499
            gradual_bal_param = 1.0 - (current_epoch / epoch_nums) * (1.0 - self.bal_param)
            ul_mask_abc = torch.bernoulli(gradual_bal_param[pred_class_abc].detach()).float()

            # mask consistency loss with soft pseudo-label
            abc_mask = loss_weight_abc * ul_mask_abc
            cons_loss_abc = -1 * torch.mean(
                abc_mask * torch.sum(p_abc * F.log_softmax(logits_strong_abc, dim=1), dim=1)
            )

            abc_loss = cls_loss_abc + cons_loss_abc
            loss_dict.update({"loss_abc": abc_loss})

            # loss computation for SSL learner.
            logits_concat = self.model.classify(feats_concat)
            l_logits = logits_concat[:num_labels]
            cls_loss = self.l_loss(l_logits, labels)
            loss_dict.update({"loss_cls": cls_loss})

            # unlabeled loss
            logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
            logits_weak = self.alpha * (logits_weak - self.pi.log())
            p = logits_weak.detach().softmax(dim=1)  # soft pseudo labels
            with torch.no_grad():
                confidence, pred_class = torch.max(p, dim=1)
            
            loss_weight = confidence
            for kbatch in range(len(confidence)):
                loss_weight[kbatch] = confidence[kbatch].ge(self.conf_thres[pred_class[kbatch].int()]).float()

            cons_loss = self.ul_loss(
                logits_strong, pred_class, weight=loss_weight, avg_factor=ul_weak.size(0)
            )
            loss_dict.update({"loss_cons": cons_loss})

        else:

            # input concatenation
            input_concat = torch.cat([l_images, ul_weak, ul_strong], 0)

            # predictions
            logits_concat = self.model(input_concat)

            # loss computation
            l_logits = logits_concat[:num_labels]

            # logit adjustment in train-time.
            if self.with_la:
                l_logits += (self.tau * self.p_data.view(1, -1).log())
            
            cls_loss = self.l_loss(l_logits, labels)
            loss_dict.update({"loss_cls": cls_loss})

            # unlabeled loss
            logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)

            # seval_pi
            logits_weak = self.alpha * (logits_weak - self.pi.log())

            p = logits_weak.detach().softmax(dim=1)  # soft pseudo labels

            if self.with_align:
                p = self.dist_align(p)  # distribution alignment

            with torch.no_grad():
                if self.with_darp:        
                    p = self.darp_optimizer.step(p, ul_indices)
                # final pseudo-labels with confidence
                confidence, pred_class = torch.max(p, dim=1)

            loss_weight = confidence
            for kbatch in range(len(confidence)):
                loss_weight[kbatch] = confidence[kbatch].ge(self.conf_thres[pred_class[kbatch].int()]).float()
            
            cons_loss = self.ul_loss(
                logits_strong, pred_class, weight=loss_weight, avg_factor=ul_weak.size(0)
            )
            loss_dict.update({"loss_cons": cons_loss})

            

        if self.daso:
            x = input_concat
            # apply the semantic loss.
            with torch.no_grad():
                l_feats = self.ema_model(x[:num_labels], return_features=True)
                self.queue.enqueue(l_feats.clone().detach(), labels.clone().detach())

            # feature vectors
            x = self.model(x, return_features=True)

            # initial empty assignment
            assignment = torch.Tensor([-1 for _ in range(len(UL_LABELS))]).float().to(self.device)
            if not self.pretraining:
                prototypes = self.queue.prototypes  # (K, D)
                feats_weak, feats_strong = x[num_labels:].chunk(2)  # (B, D)

                with torch.no_grad():
                    # similarity between weak features and prototypes  (B, K)
                    sim_weak = self.similarity_fn(
                        feats_weak.unsqueeze(1), prototypes.unsqueeze(0)
                    ) / self.T_proto
                    soft_target = sim_weak.softmax(dim=1)
                    assign_confidence, assignment = torch.max(soft_target.detach(), dim=1)

                # soft loss
                if self.psa_loss_weight > 0:
                    # similarity between strong features and prototypes  (B, K)
                    sim_strong = self.similarity_fn(
                        feats_strong.unsqueeze(1), prototypes.unsqueeze(0)
                    ) / self.T_proto

                    loss_assign = -1 * torch.sum(soft_target * F.log_softmax(sim_strong, dim=1),
                                                dim=1).sum() / sim_weak.size(0)
                    loss_dict.update({"loss_assign": self.psa_loss_weight * loss_assign})

            if self.iter + 1 >= self.pretrain_steps:
                self.pretraining = False

        losses = sum(loss_dict.values())
        # compute batch-wise accuracy and update metrics_dict
        top1, top5 = self.accuracy(l_logits, labels)
        metrics_dict.update(loss_dict)
        metrics_dict.update({"top1": top1, "top5": top5})

        # update params and schedule learning rates
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        if self.apply_scheduler:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]
        ema_decay = self.ema_model.update(self.model, step=self.iter, current_lr=current_lr)

        # measure iter time
        iter_time = time.perf_counter() - start

        # logging
        self.iter_timer.update(iter_time, n=l_images.size(0))
        self.meters.put_scalar(
            "misc/iter_time", self.iter_timer.avg, n=l_images.size(0), show_avg=False
        )
        self.meters.put_scalar("train/ema_decay", ema_decay, show_avg=False)
        self.meters.put_scalar("misc/data_time", data_time, n=l_images.size(0))
        self.meters.put_scalar("misc/lr", current_lr, show_avg=False)

        # make a log for accuracy and losses
        self._write_metrics(metrics_dict, n=l_images.size(0), prefix="train")
