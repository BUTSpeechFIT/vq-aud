import os
import sys
import argparse
import utils


def load_from_dir(nnet_dir, device):
    import models
    model_classname = open(os.path.join(nnet_dir, 'nnet_kind.txt')).read().strip()
    model_kind = models.get_model(model_classname)
    model = model_kind.load_from_dir(nnet_dir, map_location=device)
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true',
                        help='if specified, turns off gpu usage.')
    parser.add_argument('--upsample', action='store_true',
                        help='if specified, latent representation will be upsampled to input length')
    parser.add_argument('--utts',
                        help='if specified, only the utterances in this file will be decoded')
    parser.add_argument('data_npz',
                        help='directory that contains the document matrix')
    parser.add_argument('nnet_dir',
                        help='directory that contains the saved network files')
    parser.add_argument('outfile',
                        help='file into which to store the result files')
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

    device = torch.device(dev_name)
    _ = torch.tensor(1, device=device)
    data = np.load(args.data_npz)
    utts_to_decode = data.files
    if args.utts:
        if args.utts == '-':
            utts_to_decode = [line.strip().split()[0] for line in sys.stdin.readlines()]
        else:
            utts_to_decode = [x.strip().split()[0] for x in open(args.utts)]
    model = load_from_dir(args.nnet_dir, device)
    model.eval()
    decoded = {}
    outdir = os.path.dirname(args.outfile)
    utils.chk_mkdir(outdir)
    with torch.no_grad():
        with open(args.outfile, 'w', encoding='utf-8') as _out:
            for doc in utts_to_decode:
                x = torch.from_numpy(data[doc]).float().to(device).unsqueeze(0)
                encoded = model.encoder(x).contiguous()
                quantized, alignment = model.quantize(encoded, return_alignment=True)
                old_length = x.size(-2)
                new_length = x.size(-1)
                if old_length > new_length and args.upsample:
                    alignment = alignment.unsqueeze(0).unsqueeze(0).float()
                    alignment = torch.nn.functional.interpolate(alignment, old_length).squeeze()
                alignment = alignment.int().detach().cpu().numpy()
                decoded[doc] = alignment
                _out.write(f'{doc} {" ".join(["au_" + str(a) for a in alignment])}\n')


if __name__ == '__main__':
    main()
