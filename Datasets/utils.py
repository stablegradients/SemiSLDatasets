import random
import numpy as np
import torch
from math import floor


def SSLSplitCIFAR(targets, num_classes, N1, M1, val_size,\
                    rho_l, rho_u, inc_lb, seed=0, imb_type="exp"):
    def get_random_states():
        return random.getstate(), torch.get_rng_state(), np.random.get_state()

    def set_random_states(random_state, torch_state, np_state):
        random.setstate(random_state)
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        return

    def samp_per_class(N1, M1, rho_l, rho_u, num_classes, val_size, imb_type="exp"):
        if imb_type == "exp":
            gamma_l = (rho_l)**(-1/(num_classes-1))
            lb_samp_per_class = [floor(N1 * (gamma_l**i))\
                                    for i in range(num_classes)]
            
            gamma_u = (rho_u)**(-1/(num_classes-1))
            ulb_samp_per_class = [floor(M1 * (gamma_u**i))\
                                    for i in range(num_classes)]
        else :
            lb_samp_per_class = [floor(N1 * (1/rho_l) * (1- i/num_classes))\
                                                for i in range(num_classes)]
            ulb_samp_per_class = [floor(M1 * (1/rho_u) * (1- i/num_classes))\
                                                for i in range(num_classes)]

        val_samp_per_class = [floor(val_size//num_classes)\
                                    for i in range(num_classes)]

        return lb_samp_per_class, ulb_samp_per_class, val_samp_per_class

    random_state, torch_state, np_state = get_random_states()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    lb_samp_per_class, ulb_samp_per_class, val_samp_per_class = samp_per_class(
        N1, M1, rho_l, rho_u, num_classes, val_size, imb_type
    )

    targets = np.array(targets)
    lb_idx, ulb_idx, val_idx = [], [], []

    for c in range(num_classes):
        c_targets =  np.where(targets == c)[0].tolist()
        if len(c_targets) < lb_samp_per_class[c] + ulb_samp_per_class[c] + val_samp_per_class[c]:
            raise Exception("Too few samples present")
        lb_idx = lb_idx + c_targets[0:lb_samp_per_class[c]]
        ulb_idx = ulb_idx + c_targets[lb_samp_per_class[c] :\
                                    lb_samp_per_class[c] + ulb_samp_per_class[c]]
        val_idx = val_idx + c_targets[lb_samp_per_class[c] + ulb_samp_per_class[c]:\
                                      lb_samp_per_class[c] + ulb_samp_per_class[c] + val_samp_per_class[c]]
    if inc_lb:
        ulb_idx = ulb_idx + lb_idx

    set_random_states(random_state, torch_state, np_state)
    return lb_idx, ulb_idx, val_idx


def SSLSplitSTL10(labels, num_classes, N1, val_size,\
                    rho_l, seed=0, imb_type="exp"):
    def get_random_states():
        return random.getstate(), torch.get_rng_state(), np.random.get_state()

    def set_random_states(random_state, torch_state, np_state):
        random.setstate(random_state)
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        return

    def samp_per_class(N1, rho_l, num_classes, val_size, imb_type="exp"):
        if imb_type == "exp":
            gamma_l = (rho_l)**(-1/(num_classes-1))
            lb_samp_per_class = [floor(N1 * (gamma_l**i))\
                                    for i in range(num_classes)]

        else :
            lb_samp_per_class = [floor(N1 * (1/rho_l) * (1- i/num_classes))\
                                                for i in range(num_classes)]

        val_samp_per_class = [floor(val_size//num_classes)\
                                    for i in range(num_classes)]

        return lb_samp_per_class, val_samp_per_class

    random_state, torch_state, np_state = get_random_states()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    lb_samp_per_class, val_samp_per_class = samp_per_class(
        N1, rho_l, num_classes, val_size, imb_type
    )

    labels = np.array(labels)
    lb_idx, val_idx = [], []

    for c in range(num_classes):
        c_targets =  np.where(labels == c)[0].tolist()
        if len(c_targets) < lb_samp_per_class[c] + val_samp_per_class[c]:
            raise Exception("Too few samples present")
        lb_idx = lb_idx + c_targets[0:lb_samp_per_class[c]]
        val_idx = val_idx + c_targets[lb_samp_per_class[c] :\
                                      lb_samp_per_class[c] + val_samp_per_class[c]]

    set_random_states(random_state, torch_state, np_state)
    return lb_idx, val_idx