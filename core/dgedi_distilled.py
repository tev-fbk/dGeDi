import numpy as np
import torch
import time
from core.model_dgedi import PointTransformerV3


class dgedi():

    def __init__(self, config):
        self.query_model = None
        self.normalize_features = False

        if 'query' in config:
            self.query_model = self.load_model(config['query'], config['device'])

        self.device = torch.device(config['device'])

    def load_model(self, model_config, device):
        print('loading model')
        self.normalize_features = model_config.get('normalize_features', False)
        model = PointTransformerV3(
            enable_flash=model_config.get('enable_flash', False),
            enc_channels=model_config.get('enc_channels'),
            enc_num_head=model_config.get('enc_num_head'),
            dec_channels=model_config.get('dec_channels'),
            dec_num_head=model_config.get('dec_num_head'),
            pdnorm_bn=model_config.get('pd_norm_bn'),
            pdnorm_ln=model_config.get('pd_norm_ln'),
        ).to(device)

        checkpoint = torch.load(model_config['weights_path'], map_location=device)
        print('checkpoint keys:', checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def compute_feats_in_batch_variable_length(self, pts_batch, offsets=None):
        """
        Two calling conventions:
        1) (B, N, 3) pts_batch, no offsets
        2) (sumN, 3) pts_batch + offsets of shape (B+1,)
        """
        device = self.device

        if offsets is None:
            # Fixed-size batch: (B, N, 3)
            if isinstance(pts_batch, np.ndarray):
                pts_batch = torch.from_numpy(pts_batch).float()
            pts_batch = pts_batch.to(device)

            B, N, _ = pts_batch.shape
            pts_list = []
            offset_list = []
            running_offset = 0

            for b in range(B):
                pts_b = pts_batch[b] - pts_batch[b].mean(dim=0)
                pts_list.append(pts_b)
                running_offset += N
                offset_list.append(running_offset)

            pts_cat = torch.cat(pts_list, dim=0)
            offset = torch.tensor(offset_list, dtype=torch.long, device=device)

        else:
            # Variable-size: (sumN, 3) + offsets (B+1,)
            if isinstance(pts_batch, np.ndarray):
                pts_batch = torch.from_numpy(pts_batch).float()
            if isinstance(offsets, np.ndarray):
                offsets = torch.from_numpy(offsets).long()

            pts_batch = pts_batch.to(device)
            offsets = offsets.to(device)

            pts_cat = pts_batch.clone()
            for i in range(len(offsets) - 1):
                start, end = offsets[i], offsets[i + 1]
                pts_cat[start:end] -= pts_cat[start:end].mean(dim=0)

            offset = offsets[1:]

        data_dict = {
            'feat': pts_cat,
            'coord': pts_cat,
            'offset': offset,
            'grid_size': 0.0056,
        }

        t1 = time.time()
        output = self.query_model(data_dict)
        feats_out = output.feat
        print('Forward pass time:', time.time() - t1)

        if self.normalize_features:
            print('Normalizing features')
            feats_out = feats_out / (feats_out.norm(dim=1, keepdim=True))
        else:
            print('Not normalizing features')

        return feats_out
