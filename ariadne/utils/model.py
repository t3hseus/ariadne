import torch
import glob
import os

def get_checkpoint_path(model_dir, version=None, checkpoint='latest'):
    '''Function to get checkpoint of model in given directory.
    If it is needed to use not specific, but newest version of model, it can bee found automatically
    Arguments:
        model_dir (str): directory with model checkpoints
        version (str, None by default): name of directory with needed checkpoint.
                                           If 'latest', directory with maximum change time will be used
                                           If None, only model_dir and checkpoint will be used
        checkpoint (str, 'latest' by default): name of checkpoint (with .ckpt).
                                           If 'latest', checkpoint with maximum change time will be used
    '''
    if version is not None:
        if version == 'latest':
            list_of_files = glob.glob(f"{model_dir}/*")
            version = max(list_of_files, key=os.path.getmtime).split('/')[-1]
        model_dir = model_dir+'/'+version
    if checkpoint == 'latest':
        list_of_files = glob.glob(f"{model_dir}/*.ckpt")
        checkpoint = max(list_of_files, key=os.path.getmtime).split('/')[-1]
    return f'{model_dir}/{checkpoint}'

def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['state_dict']
    real_dict = {}
    for (k, v) in model_dict.items():
        needed_key = None
        for pretr_key in pretrained_dict:
            if k in pretr_key:
                needed_key = pretr_key
                break
        assert needed_key is not None, "key %s not in pretrained_dict %r!" % (k, pretrained_dict.keys())
        real_dict[k] = pretrained_dict[needed_key]

    model.load_state_dict(real_dict)
    model.eval()
    return model