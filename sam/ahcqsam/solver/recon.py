import numpy as np
import torch
import torch.nn as nn
import logging
from utils import DataSaverHook, StopForwardException, DataSaverHook_kwargs, UnifiedDataSaverHook
from ahcqsam.quantization.quantized_module import QuantizedModule
from ahcqsam.quantization.fake_quant import LSQFakeQuantize, LSQPlusFakeQuantize, QuantizeBase, LogTransformQuantize, \
    AdaptiveGranularityQuantize, GroupLSQFakeQuantize, HybridQuantize

logger = logging.getLogger('ahcqsam')


def save_inp_oup_data(model, module, cali_data: list, store_inp=False, store_oup=False, bs: int = 32,
                      keep_gpu: bool = True):
    device = next(model.parameters()).device
    # data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    # handle = module.register_forward_hook(data_saver)
    data_saver = UnifiedDataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver, with_kwargs=True)

    cached = [[], []]
    with torch.no_grad():
        for i in range(len(cali_data)):
            # print(i,len(cali_data))
            try:
                _ = model.extract_feat(cali_data[i])
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0])
                else:
                    # try:
                    # 处理数据
                    if isinstance(data_saver.input_store, dict):
                        # print("Captured Kwargs:", data_saver.input_store.keys())
                        input_data = data_saver.input_store
                    elif isinstance(data_saver.input_store, tuple):
                        # print("Captured Args")
                        input_data = data_saver.input_store[0]
                    else:
                        print("No input captured")
                    if isinstance(input_data, dict):
                        for key in input_data:
                            input_data[key] = input_data[key].detach().cpu()
                        cached[0].append(input_data)
                    elif isinstance(input_data, tuple):
                        if len(input_data) == 3:
                            cached[0].append((input_data[0].detach().cpu(), input_data[1].detach().cpu(),
                                              input_data[2].detach().cpu()))
                        else:
                            cached[0].append((input_data[0].detach().cpu(), input_data[1].detach().cpu()))
                    else:
                        cached[0].append(input_data.cpu())
            if store_oup:
                if isinstance(data_saver.output_store, (tuple, list)):
                    new_data = []
                    for v in data_saver.output_store:
                        if isinstance(v, torch.Tensor):
                            new_data.append(v.detach() if keep_gpu else v.detach().cpu())
                        else:
                            new_data.append(v)
                    cached[1].append(list(new_data))
                else:
                    cached[1].append(data_saver.output_store.detach() if keep_gpu
                                     else data_saver.output_store.detach().detach().cpu())
    # if store_inp:
    #     cached[0] = torch.cat([x for x in cached[0]])
    # if store_oup:
    #     cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    torch.cuda.empty_cache()
    return cached


def move_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, (list, tuple)):
        new_data = [move_to_cpu(item) for item in data]
        return tuple(new_data) if isinstance(data, tuple) else new_data
    elif isinstance(data, dict):
        return {k: move_to_cpu(v) for k, v in data.items()}
    else:
        return data


def move_to_gpu(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cuda(non_blocking=True)
    elif isinstance(data, (list, tuple)):
        new_data = [move_to_gpu(item) for item in data]
        return tuple(new_data) if isinstance(data, tuple) else new_data
    elif isinstance(data, dict):
        return {k: move_to_gpu(v) for k, v in data.items()}
    else:
        return data


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 module: QuantizedModule,
                 weight: float = 1.,
                 iters: int = 20000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.,
                 reg_weight=None,
                 reg_weight_lamb=0.1
                 ):

        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(iters, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.reg_weight = reg_weight
        self.reg_weight_lamb = reg_weight_lamb

    def __call__(self, pred, tgt, multi=False):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        self.count += 1
        if multi:
            rec_loss = 0.0
            for q_item, fp_item in zip(pred, tgt):
                if isinstance(q_item, torch.Tensor) and isinstance(fp_item, torch.Tensor):
                    rec_loss += lp_loss(q_item, fp_item, p=self.p)
        else:
            rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
            w_reg_loss = 0
        else:
            round_loss = 0
            w_reg_loss = 0
            layer_len = 0
            for layer in self.module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    if self.reg_weight is None:
                        round_vals = layer.weight_fake_quant.rectified_sigmoid()
                        round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        # total_loss = w_reg_loss
        if self.count % 500 == 0:
            logger.info('Total loss:\t{:.4f} (rec:{:.4f}, round:{:.4f}, rw:{:.4f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), float(w_reg_loss), b, self.count))
        return total_loss


def lp_loss(pred, tgt, p=2.0):
    """
    loss function
    """
    return (pred - tgt).abs().pow(p).sum(1).mean()


def reconstruction(model, fp_model, module, fp_module, cali_data, config, ahcqsam_config):
    device = next(module.parameters()).device
    # get data first
    quant_inp, _ = save_inp_oup_data(model, module, cali_data, store_inp=True, store_oup=False, bs=config.batch_size,
                                     keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data(fp_model, fp_module, cali_data, store_inp=True, store_oup=True,
                                       bs=config.batch_size, keep_gpu=config.keep_gpu)
    # prepare for up or down tuning
    w_para, a_para = [], []

    # # for the bimodal block, add the gamma parameter
    # gamma_para = []
    # if hasattr(module,'gamma') and config.gamma_tune:
    #     gamma_para.append(module.gamma)

    for name, layer in module.named_modules():
        only4flag = ('only4' not in config.keys()) or (not config.only4) or (
                    config.only4 and ('k_proj' in name or 'q_proj' in name))
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
            layer.drop_prob = config.drop_prob
            if only4flag:
                if isinstance(layer, LSQFakeQuantize):
                    a_para += [layer.scale]
                if isinstance(layer, LSQPlusFakeQuantize):
                    a_para += [layer.scale]
                    a_para += [layer.zero_point]
                if isinstance(layer, AdaptiveGranularityQuantize):
                    a_para += [layer.scale]
                if isinstance(layer, LogTransformQuantize):
                    pass
                if isinstance(layer, GroupLSQFakeQuantize):
                    a_para += [layer.grouped_scales]
                if isinstance(layer, HybridQuantize):
                    a_para += [layer.scale_log]
                    a_para += [layer.scale_uni]

    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para)
    else:
        w_opt = None

    logger.info(name)
    logger.info(type(module))
    logger.info(len(a_para))

    if len(a_para) == 0 and len(w_para) == 0:
        logger.info('skip opt')
        del fp_inp, fp_oup, quant_inp
        torch.cuda.empty_cache()
        for name, layer in module.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if weight_quantizer.adaround:
                    weight_quantizer = layer.weight_fake_quant
                    layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                    weight_quantizer.adaround = False
            if isinstance(layer, QuantizeBase) and 'act_fake_quantize' in name:
                layer.drop_prob = 1.0
        return

    loss_func = LossFunction(module=module, weight=config.weight, iters=config.iters, b_range=config.b_range,
                             warm_up=config.warm_up)

    from mmdet.utils import build_ddp, build_dp
    import os
    module_ddp = build_dp(module, 'cuda', device_ids=[0])
    # module_ddp = build_ddp(
    #     module,
    #     'cuda',
    #     device_ids=[int(os.environ['LOCAL_RANK'])],
    #     broadcast_buffers=False)
    if config.keep_gpu:
        try:
            fp_oup = move_to_gpu(fp_oup)
            fp_inp = move_to_gpu(fp_inp)
            quant_inp = move_to_gpu(quant_inp)

        except:
            in_cpu = 32
            logger.info('in_cpu 32')
            fp_oup = move_to_cpu(fp_oup)
            fp_inp = move_to_cpu(fp_inp)
            quant_inp = move_to_cpu(quant_inp)
    else:
        in_cpu = 32
        logger.info('in_cpu 32')
        fp_oup = move_to_cpu(fp_oup)
        fp_inp = move_to_cpu(fp_inp)
        quant_inp = move_to_cpu(quant_inp)

    sz = len(cali_data)
    for i in range(config.iters):
        idx = torch.randint(0, sz, (1,))
        if config.drop_prob < 1.0:
            # cur_quant_inp = quant_inp[idx].to(device)
            # cur_quant_inp = quant_inp[idx]
            cur_quant_inp = quant_inp[idx]
            cur_fp_inp = fp_inp[idx]

            # cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
            if isinstance(cur_quant_inp, dict):
                cur_inp = {}
                for key in cur_quant_inp:
                    q_tensor = cur_quant_inp[key].cuda(non_blocking=True)
                    f_tensor = cur_fp_inp[key].cuda(non_blocking=True)
                    cur_inp[key] = torch.where(
                        torch.rand_like(q_tensor) < config.drop_prob,
                        q_tensor,
                        f_tensor
                    )
            elif isinstance(cur_quant_inp, torch.Tensor):
                cur_quant_inp = cur_quant_inp.cuda(non_blocking=True)
                cur_fp_inp = cur_fp_inp.cuda(non_blocking=True)
                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob,
                                      cur_quant_inp, cur_fp_inp.cuda(non_blocking=True))
            elif len(cur_quant_inp) == 2:

                cur_quant_inp = (cur_quant_inp[0].cuda(non_blocking=True), cur_quant_inp[1].cuda(non_blocking=True))
                cur_fp_inp = (cur_fp_inp[0].cuda(non_blocking=True), cur_fp_inp[1].cuda(non_blocking=True))

                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0],
                                       cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1],
                                       cur_fp_inp[1])
                cur_inp = (cur_inp0, cur_inp1)
            else:
                cur_quant_inp = (cur_quant_inp[0].cuda(non_blocking=True), cur_quant_inp[1].cuda(non_blocking=True),
                                 cur_quant_inp[2].cuda(non_blocking=True))
                cur_fp_inp = (cur_fp_inp[0].cuda(non_blocking=True), cur_fp_inp[1].cuda(non_blocking=True),
                              cur_fp_inp[2].cuda(non_blocking=True))

                cur_inp0 = torch.where(torch.rand_like(cur_quant_inp[0]) < config.drop_prob, cur_quant_inp[0],
                                       cur_fp_inp[0])
                cur_inp1 = torch.where(torch.rand_like(cur_quant_inp[1]) < config.drop_prob, cur_quant_inp[1],
                                       cur_fp_inp[1])
                cur_inp2 = torch.where(torch.rand_like(cur_quant_inp[2]) < config.drop_prob, cur_quant_inp[2],
                                       cur_fp_inp[2])
                cur_inp = (cur_inp0, cur_inp1, cur_inp2)
        else:
            cur_inp = quant_inp[idx]

        cur_fp_oup = move_to_gpu(fp_oup[idx])
        if a_opt:
            a_opt.zero_grad()
        # if gamma_opt:
        #     gamma_opt.zero_grad()
        if w_opt:
            w_opt.zero_grad()
        # import pdb;pdb.set_trace()
        if isinstance(cur_inp, dict):
            cur_quant_oup = module_ddp(**cur_inp)
        else:
            cur_quant_oup = module_ddp(cur_inp)
        if isinstance(cur_quant_oup, (tuple, list)):
            err = 0.0
            for q_item, fp_item in zip(cur_quant_oup, cur_fp_oup):
                if isinstance(q_item, torch.Tensor) and isinstance(fp_item, torch.Tensor):
                    err += loss_func(q_item, fp_item)
            # err = loss_func(cur_quant_oup, cur_fp_oup, multi=True)


        else:
            err = loss_func(cur_quant_oup, cur_fp_oup)
        cur_inp = None
        cur_quant_oup = None
        # torch.cuda.empty_cache()
        err.backward()  # del cur_inp cur_quant_oup
        if w_opt:
            w_opt.step()
        # if gamma_opt:
        #     gamma_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()

        if ahcqsam_config.cag:
            contains_group_lsq = any(isinstance(sub_module, GroupLSQFakeQuantize) for sub_module in module.modules())
            if contains_group_lsq:
                if i in [int(config.iters * 0.2), int(config.iters * 0.4), int(config.iters * 0.6),
                         int(config.iters * 0.8)]:
                    if i == int(config.iters * 0.2):
                        a_opt, a_scheduler = group_channel(module, a_para, config, num_channel=ahcqsam_config.group * 8)
                    if i == int(config.iters * 0.4):
                        a_opt, a_scheduler = group_channel(module, a_para, config, num_channel=ahcqsam_config.group * 4)
                    if i == int(config.iters * 0.6):
                        a_opt, a_scheduler = group_channel(module, a_para, config, num_channel=ahcqsam_config.group * 2)
                    if i == int(config.iters * 0.8):
                        a_opt, a_scheduler = group_channel(module, a_para, config, num_channel=ahcqsam_config.group)

    del fp_inp, fp_oup, quant_inp, cur_fp_oup
    torch.cuda.empty_cache()

    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if weight_quantizer.adaround:
                weight_quantizer = layer.weight_fake_quant
                layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
                weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0


def group_channel(module, a_para, config, num_channel):
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, GroupLSQFakeQuantize):
            sub_module.group_channel(num_channel)
            logger.info(f'Group number of activation channel into {num_channel}')
            a_para += [sub_module.grouped_scales]
            if len(a_para) != 0:
                a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
                a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
            else:
                a_opt, a_scheduler = None, None
    return a_opt, a_scheduler


import torch
import torch.nn as nn
import math
import logging


def condiBasedAct_reconstruction(model, fp_model, module, fp_module, cali_data, config, ahcqsam_config):
    device = next(module.parameters()).device
    # get data first
    quant_inp, _ = save_inp_oup_data(model, module, cali_data, store_inp=True, store_oup=False, bs=config.batch_size,
                                     keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data(fp_model, fp_module, cali_data, store_inp=True, store_oup=True,
                                       bs=config.batch_size, keep_gpu=config.keep_gpu)
    # prepare for up or down tuning
    w_para, a_para = [], []

    loss_func = LossFunction(module=module, weight=config.weight, iters=config.iters, b_range=config.b_range,
                             warm_up=config.warm_up)

    if not module.layer_pre_act_fake_quantize.fake_quant_enabled or len(module.module.weight.shape) != 2:
        return 0, 0

    from mmdet.utils import build_ddp, build_dp
    import os
    module_ddp = build_dp(module, 'cuda', device_ids=[0])
    try:
        for i in range(len(fp_oup)):
            fp_oup[i] = fp_oup[i].cuda(non_blocking=True)

        for i, t in enumerate(fp_inp):
            if isinstance(t, tuple):
                if len(t) == 3:
                    fp_inp[i] = (
                    t[0].cuda(non_blocking=True), t[1].cuda(non_blocking=True), t[2].cuda(non_blocking=True))
                else:
                    fp_inp[i] = (t[0].cuda(non_blocking=True), t[1].cuda(non_blocking=True))
            else:
                fp_inp[i] = t.cuda(non_blocking=True)

        for i, t in enumerate(quant_inp):
            if isinstance(t, tuple):
                if len(t) == 3:
                    quant_inp[i] = (
                    t[0].cuda(non_blocking=True), t[1].cuda(non_blocking=True), t[2].cuda(non_blocking=True))
                else:
                    quant_inp[i] = (t[0].cuda(non_blocking=True), t[1].cuda(non_blocking=True))
            else:
                quant_inp[i] = t.cuda(non_blocking=True)
    except:
        in_cpu = 32
        logger.info('in_cpu 32')
        for i in range(len(fp_oup)):
            fp_oup[i] = fp_oup[i].cpu()

        for i, t in enumerate(fp_inp):
            if i < in_cpu:
                if isinstance(t, tuple):
                    if len(t) == 3:
                        fp_inp[i] = (t[0].cpu(), t[1].cpu(), t[2].cpu())
                    else:
                        fp_inp[i] = (t[0].cpu(), t[1].cpu())
                else:
                    fp_inp[i] = t.cpu()
            else:
                if isinstance(t, tuple):
                    if len(t) == 3:
                        fp_inp[i] = (
                        t[0].cuda(non_blocking=True), t[1].cuda(non_blocking=True), t[2].cuda(non_blocking=True))
                    else:
                        fp_inp[i] = (t[0].cuda(non_blocking=True), t[1].cuda(non_blocking=True))
                else:
                    fp_inp[i] = t.cuda(non_blocking=True)

        for i, t in enumerate(quant_inp):
            if i < in_cpu:
                if isinstance(t, tuple):
                    if len(t) == 3:
                        quant_inp[i] = (t[0].cpu(), t[1].cpu(), t[2].cpu())
                    else:
                        quant_inp[i] = (t[0].cpu(), t[1].cpu())
                else:
                    quant_inp[i] = t.cpu()
            else:
                if isinstance(t, tuple):
                    if len(t) == 3:
                        quant_inp[i] = (
                        t[0].cuda(non_blocking=True), t[1].cuda(non_blocking=True), t[2].cuda(non_blocking=True))
                    else:
                        quant_inp[i] = (t[0].cuda(non_blocking=True), t[1].cuda(non_blocking=True))
                else:
                    quant_inp[i] = t.cuda(non_blocking=True)

    def _proximal_step_inline_stable(W_in, C_i_stable):

        with torch.no_grad():
            try:
                U, S, Vh = torch.linalg.svd(W_in, full_matrices=False)
            except torch.linalg.LinAlgError:
                logger.warning("SVD does not converge, use previous W_k.")
                return W_in

            if t_mode == 'mean':
                t = torch.mean(S)
            elif t_mode == 'median':
                t = torch.median(S)
            else:
                t = (torch.max(S) + torch.min(S)) / 2

            sigma_i = S
            numerator = sigma_i + 2 * lambda_ * t

            denominator = 1 + (2 * lambda_) + (2 * beta * C_i_stable)

            sigma_i_formula = numerator / denominator

            energies = S ** 2
            total_energy = torch.sum(energies)

            if total_energy == 0:
                return W_in

            c_energies = torch.cumsum(energies, dim=0)

            energy_threshold = 0.8 * total_energy

            indices_over_threshold = torch.where(c_energies > energy_threshold)[0]

            if len(indices_over_threshold) == 0:
                cutoff_index = len(S) - 1
            else:
                cutoff_index = indices_over_threshold[0]

            sigma_i_final_output = S.clone()

            minor_targets = torch.maximum(
                S[cutoff_index + 1:],
                sigma_i_formula[cutoff_index + 1:]
            )

            sigma_i_final_output[cutoff_index + 1:] = minor_targets

            W_out = U @ torch.diag(sigma_i_final_output) @ Vh
            return W_out

    w_opt = None
    w_para = []
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            w_para += [layer.weight]
    w_opt = torch.optim.Adam(w_para, lr=0.0000001)
    lambda_ = 0.003
    beta = 0.001
    t_mode = 'mean'
    cond_num_before_optimization = cond_num_after_optimization = 0
    with torch.no_grad():
        try:
            U, S, Vh = torch.linalg.svd(module.module.weight.data.clone().T, full_matrices=False)
            s_max = torch.max(S)
            s_min = torch.min(S)
            cond_num_before_optimization = s_max / s_min
            logger.info(f"Before Optimization \
                        Weight's Max: {s_max}, Min: {s_min}, Condi Number: {cond_num_before_optimization}")
        except torch.linalg.LinAlgError:
            logger.warning("SVD does not converge.")

    if cond_num_before_optimization < 100:
        logger.info(f"Skip this layer due to the Condi Number {cond_num_before_optimization} < 100")
        return 0, 0

    accumulation_steps = 32
    W_ori = module.module.weight.data.clone()
    K = ahcqsam_config.k
    logger.warning("Use Single Sample Per-Time")
    logger.info(f"Use Gradient Accumulation: Batch Size = {accumulation_steps}.")

    for i in range(K):

        total_loss = 0.0

        with torch.no_grad():

            W_in = module.module.weight.data.clone().T
            try:
                U, S, Vh = torch.linalg.svd(W_in, full_matrices=False)
            except torch.linalg.LinAlgError:
                logger.warning("Skip because SVD does not converge.")
                continue

            C_i_sum = 0.0
            total_samples_processed = 0
            for prox_step in range(accumulation_steps):
                idx = prox_step
                cur_inp_p = quant_inp[idx].to(device)
                cur_fp_inp_p = fp_inp[idx].to(device)

                delta_X_batch = cur_fp_inp_p - module.layer_pre_act_fake_quantize(cur_inp_p)

                assert delta_X_batch.shape[-1] == module.module.in_features
                D_in = module.module.in_features
                delta_X_r = delta_X_batch.reshape(-1, D_in).to(module.module.weight.device)

                delta_X_tilde = delta_X_r @ U
                C_i_current_sum = torch.sum(delta_X_tilde ** 2, dim=0)
                C_i_sum += C_i_current_sum
                num_samples_in_batch = delta_X_r.shape[0]
                total_samples_processed += num_samples_in_batch

                C_i_sum += C_i_current_sum

            C_i_stable = C_i_sum / total_samples_processed

            W_plus = _proximal_step_inline_stable(W_in, C_i_stable)

            module.module.weight.data = W_plus.T.data

        if (i + 1) % 10 == 0:
            avg_loss = total_loss / accumulation_steps
            logger.info(f"PGD Iteration [{i + 1}/{K}] Avg Loss (Eff. BS={accumulation_steps}): {avg_loss:.4f}")

    with torch.no_grad():
        try:
            U, S, Vh = torch.linalg.svd(W_in, full_matrices=False)
            s_max = torch.max(S)
            s_min = torch.min(S)
            cond_num_after_optimization = s_max / s_min
            logger.info(f"after optimization \
                        weight's max: {s_max}, min: {s_min}, condi number: {cond_num_after_optimization}")
        except torch.linalg.LinAlgError:
            logger.warning("SVD does not converge.")
    if K > 0:
        del fp_inp, fp_oup, quant_inp
    torch.cuda.empty_cache()
    return cond_num_before_optimization, cond_num_after_optimization
