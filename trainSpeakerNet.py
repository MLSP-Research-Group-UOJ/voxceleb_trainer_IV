#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp

warnings.simplefilter("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #Added by Anuraj
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Added by Anuraj
## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description="SpeakerNet");

parser.add_argument('--config', type=str, default=None, help='Config YAML file');

## Data loader
parser.add_argument('--max_frames', type=int, default=200, help='Input length to the network for training');
parser.add_argument('--eval_frames', type=int, default=0,
                    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size', type=int, default=200, help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int, default=500,
                    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=5, help='Number of loader threads');
parser.add_argument('--augment', type=bool, default=False, help='Augment input')
parser.add_argument('--seed', type=int, default=10, help='Seed for the random number generator');

## Training details
parser.add_argument('--test_interval', type=int, default=1, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epochs');
parser.add_argument('--trainfunc', type=str, default="softmaxproto", help='Loss function');

## Optimizer
parser.add_argument('--optimizer', type=str, default="adam", help='sgd or adam');
parser.add_argument('--scheduler', type=str, default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay in the optimizer');

## Loss functions
parser.add_argument("--hard_prob", type=float, default=0.5,
                    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank", type=int, default=10,
                    help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin', type=float, default=0.1, help='Loss margin, only for some loss functions');
parser.add_argument('--scale', type=float, default=30, help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker', type=int, default=1,
                    help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses', type=int, default=5994,
                    help='Number of speakers in the softmax layer, only for softmax-based losses');

## Evaluation parameters
parser.add_argument('--dcf_p_target', type=float, default=0.05,
                    help='A priori probability of the specified target speaker');
parser.add_argument('--dcf_c_miss', type=float, default=1, help='Cost of a missed detection');
parser.add_argument('--dcf_c_fa', type=float, default=1, help='Cost of a spurious detection');

## Load and save
parser.add_argument('--initial_model', type=str,
                    default="/mnt/ricproject4/commercial_product/speaker_verification/anuraj/voxceleb_trainer_10/voxceleb_trainer/initial/model/baseline_v2_ap_2022_03_20_V1.model",
                    help='Initial model weights')
parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path for model and logs')

## Training and test data

# Updated training data locations
# training english
parser.add_argument('--train_list', type=str, default="lists/train_list_Voxceleb2.txt", help='Train list')
parser.add_argument('--train_path', type=str, default="data/voxceleb2", help='Absolute path to the train set')

##training Tamil
# parser.add_argument('--train_list',     type=str,   default="lists/train_list_tamil.txt",  help='Train list')
# parser.add_argument('--train_path',     type=str,   default="data/Tamil_train", help='Absolute path to the train set')
#
##testing tamil
# parser.add_argument('--test_path',     type=str,   default="data/Tamil_test",  help='test path')
# parser.add_argument('--test_list',     type=str,   default="lists/test_list_tamil.txt",   help='Evaluation list')
#
##testing Sinhala
# parser.add_argument('--test_path',     type=str,   default="data/Sinhala_test",  help='test path')
# parser.add_argument('--test_list',     type=str,   default="lists/test_list_sinhala.txt",   help='Evaluation list')

# testing English
parser.add_argument('--test_path', type=str, default="data/voxceleb1", help='test path')
parser.add_argument('--test_list', type=str, default="lists/VoxCeleb1_test.txt", help='Evaluation list')

# Removed data augmentaions
parser.add_argument('--musan_path', type=str, default="", help='Absolute path to the test set')
parser.add_argument('--rir_path', type=str, default="", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels', type=int, default=64,
                    help='Number of mel filterbanks')  # Change the nmels from 128 to 80
parser.add_argument('--log_input', type=bool, default=True, help='Log input features')
parser.add_argument('--model', type=str, default="ResNetSE34V2", help='Name of model definition')
parser.add_argument('--encoder_type', type=str, default="ASP", help='Type of encoder')
parser.add_argument('--nOut', type=int, default=512, help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride', type=int, default=10,
                    help='Stride size of the first analytic filterbank layer of RawNet3')

## For test only
parser.add_argument('--eval', dest='eval', action='store_false', help='Eval only')

## For xvector extraction only - added by @dimuthuanuraj
parser.add_argument('--extract', dest='extract', action='store_true', help='For xvector extraction');

## Distributed and mixed precision training
parser.add_argument('--port', type=str, default="8888", help='Port for distributed training, input as text');
parser.add_argument('--distributed', dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec', dest='mixedprec', action='store_true', help='Enable mixed precision training')

args = parser.parse_args();


## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
            return opt.type
    raise ValueError


if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ## Load models
    s = SpeakerNet(**vars(args));

    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    eers = [100];

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile = open(args.result_save_path + "/scores.txt", "a+");

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))

    train_sampler = train_dataset_sampler(train_dataset, **vars(args))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    # trainLoader = get_data_loader(args.train_list, **vars(args));
    trainer = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model' % args.model_save_path)
    modelfiles.sort()

    if (args.initial_model != ""):
        trainer.loadParameters(args.initial_model);
        print("Model {} loaded!".format(args.initial_model));
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1]);
        print("Model {} loaded from previous state!".format(modelfiles[-1]));
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    for ii in range(1, it):
        trainer.__scheduler__.step()

    ## Evaluation code - must run on single GPU
    if args.eval == True:

        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        print('Total parameters: ', pytorch_total_params)
        print('Test list', args.test_list)

        # sc, lab, _ = trainer.evaluateFromList(**vars(args)) #Orginal line
        sc, lab, trials = trainer.evaluateFromList(
            **vars(args))  # modified to extract the score values for each speaker

        # Ask to save the soucre file, if so then run the following code
        print('\nDo you want to save scores Type - Y. Otherwise, Type - N.')
        userin = input()
        if (userin in ['y', 'Y', 'yes', 'Yes', 'YES']):

            # Added by Anuraj
            # https://github.com/clovaai/voxceleb_trainer/commit/4dfc8335a5e5201fffc5a7b4f59740f892cbab1a

            ## Evaluation code
            # sc, lab, trials = s.evaluateFromList(args.test_list, print_interval=100, test_path=args.test_path, eval_frames=args.eval_frames)
            # result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
            # print('\n EER %2.4f'%result[1])

            ## Save scores
            print('Type desired file name to save scores.')
            userinp = input()

            while True:
                if userinp == '':
                    print('%s Please enter the path.' % (userinp))
                    userinp = input()
                elif os.path.exists(userinp):
                    print('%s already exists. Try again.' % (userinp))
                    userinp = input()
                else:
                    with open(userinp, 'w') as outfile:
                        for vi, val in enumerate(sc):
                            outfile.write('%.4f %s\n' % (val, trials[vi]))
                    quit();

            ################

        elif (userin in ['n', 'N', 'no', 'No', 'NO']):
            print('Exiting Program without saving the score file');
        if args.gpu == 0:
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
            print("\n Threshold {:2.5f}".format(threshold));
            print('\n', time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]),
                  "MinDCF {:2.5f}".format(mindcf))

        return

    ## x-vector extraction - by dimuthuanuraj
    if args.extract == True:
        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        print('Total parameters: ', pytorch_total_params)
        print('Test list', args.test_list)
        print('Extracting embeddings from Test list', args.test_list)

        trainer.extractXvectors(**vars(args))

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path + '/run%s.zip' % strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd' % strtime, 'w') as f:
            f.write('%s' % args)

    ## Core training script
    for it in range(it, args.max_epoch + 1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0));

        if args.gpu == 0:
            print('\n', time.strftime("%Y-%m-%d %H:%M:%S"),
                  "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, max(clr)));
            scorefile.write(
                "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, max(clr)));

        if it % args.test_interval == 0:

            sc, lab, _ = trainer.evaluateFromList(**vars(args))

            if args.gpu == 0:
                result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss,
                                                  args.dcf_c_fa)

                eers.append(result[1])

                print('\n', time.strftime("%Y-%m-%d %H:%M:%S"),
                      "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf));
                scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf));

                trainer.saveParameters(args.model_save_path + "/model%09d.model" % it);

                with open(args.model_save_path + "/model%09d.eer" % it, 'w') as eerfile:
                    eerfile.write('{:2.4f}'.format(result[1]))

                scorefile.flush()

    if args.gpu == 0:
        scorefile.close();


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path = args.save_path + "/model"
    args.result_save_path = args.save_path + "/result"
    args.feat_save_path = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:', args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()