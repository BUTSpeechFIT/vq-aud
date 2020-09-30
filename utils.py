import os
import pickle
import sys
import shlex
import subprocess
import time


class SimpleLoader:
    def initialize_args(self, kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def since(t0):
    return time.time() - t0


def reduce_lr(history, lr, cooldown=0, patience=5, mode='min',
              difference=0.001, lr_scale=0.5, lr_min=0.00001,
              cool_down_patience=None):
    if cool_down_patience and cooldown <= cool_down_patience:
        return lr, cooldown+1
    assert lr_scale < 1
    if mode == 'max':
        h = [-a for a in history]
    else:
        h = history
    history = h
    len_hist = len(history)
    if len_hist <= patience:
        return lr, cooldown + 1
    recent_history = history[len_hist-patience:len_hist]
    antiquity = history[0:len_hist-patience]
    ma = min(recent_history)
    jc = min(antiquity)

    if jc - ma >= difference:
        return lr, cooldown+1
    else:
        return max(lr*lr_scale, lr_min), 0


def stop_early(history, patience=5, mode='min', difference=0.001):
    if mode == 'max':
        h = [-a for a in history]
    else:
        h = history
    history = h
    len_hist = len(history)
    if len_hist <= patience:
        return False
    recent_history = history[len_hist-patience:len_hist]
    antiquity = history[0:len_hist-patience]
    ma = min(recent_history)
    jc = min(antiquity)

    if jc - ma >= difference:
        return False
    else:
        return True


def chk_mkdir(dirname):
    if isinstance(dirname, str):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
    else:
        try:
            dirnames = iter(dirname)
            for d in dirnames:
                chk_mkdir(d)
        except TypeError:
            if not os.path.isdir(dirname):
                os.makedirs(dirname)


def pkl_save(obj, filename):
    outdir = os.path.dirname(filename)
    chk_mkdir(outdir)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pkl_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_free_gpus():
    """ Get Free GPUs.

    Returns a list of free GPU IDs
    """

    cmd = "nvidia-smi -L"

    args = shlex.split(cmd)

    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, _ = proc.communicate()
    out = out.decode('ascii').strip()

    num_gpus = len(out.split("\n"))

    free_gpus = []

    for gpu_id in range(num_gpus):

        cmd = "nvidia-smi -q -i " + str(gpu_id)
        args = shlex.split(cmd)

        proc1 = subprocess.Popen(args, stdout=subprocess.PIPE)
        proc2 = subprocess.Popen(["grep", "Process ID"],
                                 stdin=proc1.stdout,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        out, _ = proc2.communicate()

        out = out.decode('ascii').strip()
        # err = err.decode('ascii').strip()

        if not out:
            free_gpus.append(gpu_id)

    if not free_gpus:
        print("* No free GPUs found.")
        sys.exit()

    return free_gpus


def get_free_gpu_str(n=1, sep=','):
    free_gpus = [str(i) for i in get_free_gpus()]
    n = min(n, len(free_gpus))
    free_gpus = free_gpus[:n]
    return sep.join(free_gpus)


def str2float2int(x):
    """
    :param x:
    :return: x as an int
    mostly a convenience function for argparse of large numbers e.g. "1e10"
    """
    return int(float(x))
