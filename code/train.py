"""
Train one model from a config.

    python train.py -opt configs/arcgis_A.yml
    ... --debug                      (tiny run: log every iter, validate + checkpoint every 8)
    ... --auto_resume                (continue from experiments/<name>/training_states)
    ... --force_yml train:total_iter=3000 name=my_run

Outputs go to experiments/<name>/ (models, training_states, log) and two CSVs that
evaluate.py plots: experiments/<name>/losses.csv (every logged value, every print_freq iterations) and
experiments/<name>/val.csv (every validation metric).
"""
import os
import sys

import bootstrap  # noqa: F401  (repo root as working directory, .env, basicsr compatibility)
import csv
import time
import logging
import datetime
import warnings

import torch
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.train import load_resume_state, create_train_val_dataloader, init_tb_loggers
from basicsr.utils import AvgTimer, MessageLogger, get_env_info, get_root_logger, get_time_str, make_exp_dirs, \
    mkdir_and_rename
from basicsr.utils.options import copy_opt_file, dict2str

import esrgan  # noqa: F401  (registers network, dataset, model, metrics)
from esrgan.options import parse_options

# basicsr triggers torchvision deprecation warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def csv_append(path, row):
    """Append a row; the header grows if a later row has new columns (e.g. l_g_pix only after the D warm-up)."""
    rows = []
    if os.path.exists(path):
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def record_validation(model, opt, epoch, current_iter):
    results = getattr(model, 'metric_results', None)
    if results:
        csv_append(os.path.join(opt['path']['experiments_root'], 'val.csv'),
                   {'iter': current_iter, 'epoch': epoch, **{k: round(float(v), 5) for k, v in results.items()}})


def train_pipeline(root_path):
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    torch.backends.cudnn.benchmark = True

    resume_state = load_resume_state(opt)
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(os.path.join(root_path, 'tb_logger', opt['name']))
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_file = os.path.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    train_loader, train_sampler, val_loaders, total_epochs, total_iters = create_train_val_dataloader(opt, logger)

    model = build_model(opt)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch, current_iter = resume_state['epoch'], resume_state['iter']
    else:
        start_epoch, current_iter = 0, 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode in (None, 'cpu'):
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
        prefetcher = CUDAPrefetcher(train_loader, opt)
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters:
                break

            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                msg_logger.reset_start_time()

            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'lrs': model.get_current_learning_rate(),
                            'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()}
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                csv_append(os.path.join(opt['path']['experiments_root'], 'losses.csv'),
                           {'iter': current_iter, 'epoch': epoch, 'lr': model.get_current_learning_rate()[0],
                            'sec_per_iter': round(iter_timer.get_avg_time(), 4),
                            **{k: round(float(v), 6) for k, v in model.get_current_log().items()}})

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt.get('val') is not None and current_iter % opt['val']['val_freq'] == 0:
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                    record_validation(model, opt, epoch, current_iter)

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 = latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
            record_validation(model, opt, epoch, current_iter)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    train_pipeline(os.path.dirname(os.path.abspath(__file__)))
