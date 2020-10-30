import argparse
import os
import utils
import json


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='training vector quantized neural networks for AUD')
    parser.add_argument('--no-cuda', action='store_true',
                        help='if specified, turns off gpu usage.')
    parser.add_argument('--resume', action='store_true',
                        help='resume from previous checkpoint')
    parser.add_argument('--num-workers', '--nw', type=int, default=1,
                        help='number of jobs for loading data')
    parser.add_argument('--batch-size', '-b', default=32, type=int,
                        help='batch size')
    parser.add_argument('--job-num-epochs', '--nj', type=int, default=1,
                        help='number of training epochs for a single job')
    parser.add_argument('--trainer', '--tc', default='Simple',
                        help='trainer name as defined in training.py')
    parser.add_argument('--trainer-json', '--tj', '--tconf',
                        help='path to trainer config jsonfile')
    parser.add_argument('--validation-split', '--spl', type=float, default=0.2,
                        help='ratio of data to use of validation')
    parser.add_argument('--dataset-name', '--dtn', default='SimpleDataset',
                        help='dataset name as defined in dataloaders.py')
    parser.add_argument('--model', default='VQVAE',
                        help='model structure as defined in models.py')
    parser.add_argument('--num-centroids', '-n', default=512, type=int,
                        help='number of centroids in the vector quantizer')
    parser.add_argument('--rec-loss-weight', '-r', type=float, default=1,
                        help='weight of the reconstruction loss')
    parser.add_argument('--dis-loss-weight', '-d', type=float, default=1,
                        help='weight of the embedding distance loss')
    parser.add_argument('--com-loss-weight', '-c', type=float, default=1,
                        help='weight of the commitment loss')
    parser.add_argument('--utt2spk', '-u',
                        help='file mapping utterances to speaker IDs, also doubles as a '
                             'speaker embedding training flag')
    parser.add_argument('--use-ma', action='store_true',
                        help='use exponential moving average to update the centroids instead of gradient descent')
    parser.add_argument('--ma-momentum', type=float, default=0.9,
                        help='momentum to use for exponential moving average')
    parser.add_argument('--speaker-embeddings-dim', '-s', type=int, default=32,
                        help='dimension of speaker embeddings, only useful if utt2spk is set')
    parser.add_argument('encoder_json',
                        help='json file defining the encoder model structure as defined in layers.py')
    parser.add_argument('decoder_json',
                        help='json file defining the decoder model structure as defined in layers.py')
    parser.add_argument('data_npz',
                        help='numpy npz containing the features')
    parser.add_argument('output_directory', help='Model output directory')

    args = parser.parse_args()
    if args.no_cuda:
        dev_name = 'cpu'
    else:
        cuda_id = utils.get_free_gpu_str(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
        dev_name = 'cuda:0'

    import numpy as np
    import torch
    import models
    import training
    import losses
    import dataloaders

    outdir = args.output_directory
    device = torch.device(dev_name)
    _ = torch.tensor(1, device=device)
    data = np.load(args.data_npz)

    featdim = data[data.files[0]].shape[1]
    docs = data.files
    # len_val = int(args.validation_split * len(docs))
    # train_docs = docs[len_val:]
    train_docs = docs[:]
    # val_docs = docs[:len_val]
    val_docs = docs[:]
    pad_value = -1000

    if args.utt2spk:
        utt2spk = {x.strip().split()[0]: x.strip().split()[1]
                   for x in open(args.utt2spk)}
        speakers = sorted(list(set([x for x in utt2spk.values()])))
        speakers = {x: i for i, x in enumerate(speakers)}
        num_speakers = len(speakers)
    else:
        utt2spk = None
        speakers = None
        num_speakers = None
    dataset_class = dataloaders.get_dataset(args.dataset_name)
    train_dataloader = dataloaders.dataloader(dataset_class,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=False,
                                              pin_memory=True,
                                              pad_value=pad_value,
                                              data=data,
                                              utt2spk=utt2spk,
                                              speakers=speakers,
                                              docs=train_docs,
                                              sort_by_length=True,
                                              )
    valid_dataloader = dataloaders.dataloader(dataset_class,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=False,
                                              pin_memory=True,
                                              pad_value=pad_value,
                                              data=data,
                                              utt2spk=utt2spk,
                                              speakers=speakers,
                                              docs=val_docs,
                                              sort_by_length=True,
                                              )

    data_loaders = {'train': train_dataloader, 'test': valid_dataloader}

    encoder_model = models.CompositeModel(args.encoder_json,
                                          input_dim=featdim)
    decoder_model = models.CompositeModel(args.decoder_json,
                                          input_dim=encoder_model.output_dim)

    vq_model = models.VQVAE(encoder=encoder_model, decoder=decoder_model,
                            num_centroids=args.num_centroids,
                            num_speakers=num_speakers,
                            speaker_embeddings_dim=args.speaker_embeddings_dim,
                            use_ma=args.use_ma,
                            ma_momentum=args.ma_momentum)

    trainer_class = training.get_trainer(args.trainer)
    trainer_conf = {}
    if args.trainer_json:
        with open(args.trainer_json) as _json:
            trainer_conf = json.load(_json)
    if args.resume:
        training_loop = trainer_class(outdir, device)
    else:
        training_loop = trainer_class(vq_model, device, **trainer_conf)
    loss_class = losses.MaskedMSELoss
    criteria = {'rec_loss': loss_class(pad_value=pad_value, weight=args.rec_loss_weight),
                'dis_loss': loss_class(pad_value=pad_value, weight=args.dis_loss_weight),
                'com_loss': loss_class(pad_value=pad_value, weight=args.com_loss_weight),
                }
    training_loop.train(outdir, criteria, data_loaders=data_loaders,
                        phases=['train', 'test'],
                        job_num_epochs=args.job_num_epochs)


if __name__ == '__main__':
    main()
