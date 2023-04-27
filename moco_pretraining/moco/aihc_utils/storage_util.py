import os
import datetime

from pathlib import Path
import getpass




def get_storage_folder(exp_name, exp_type,exp_storage):

    if str(getpass.getuser()) == 'jby':
        STORAGE_ROOT = Path('/home/jby/chexpert_experiments')
    else:
        #STORAGE_ROOT = Path('/deep/group/aihc-bootcamp-spring2020/cxr_fewer_samples/experiments')
        STORAGE_ROOT = Path(exp_storage)

    try:
        jobid = os.environ["SLURM_JOB_ID"]
    except:
        jobid = None

    datestr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    username = str(getpass.getuser())

    fname = f'{exp_name}_{exp_type}_{datestr}_SLURM{jobid}' if jobid is not None else f'{exp_name}_{exp_type}_{datestr}'

    path_name = STORAGE_ROOT / username / fname

    try:
        original_umask = os.umask(0)
        os.makedirs(path_name, 0o777)
    finally:
        os.umask(original_umask)

    print(f'Experiment storage is at {fname}')
    return path_name