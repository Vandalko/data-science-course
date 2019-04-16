import os
import json
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.resnet as module_arch
from train import get_instance
import numpy as np
import pandas as pd


def main(config, resume):
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['csv_path'],
        img_size=config['data_loader']['args']['img_size'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=0
    )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    sample_submission = pd.read_csv(config['input_csv'])

    os.makedirs("./submit", exist_ok=True)

    thresholds = [0.4, 0.5]
    for threshold in thresholds:
        filenames, labels, submissions = [], [], []
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data = data.to(device)
                output = model(data)
                label = output.sigmoid().cpu().data.numpy()

                filenames.append(target)
                labels.append(label > threshold)

        for row in np.concatenate(labels):
            subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
            submissions.append(subrow)
        sample_submission['Predicted'] = submissions
        sample_submission.to_csv("./submit/submission-{0:.2f}.csv".format(threshold), index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.config:
        config = json.load(open(args.config))
    elif args.resume:
        config = torch.load(args.resume)['config']

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
