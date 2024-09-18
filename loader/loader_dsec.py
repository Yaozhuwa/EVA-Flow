from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
import numpy as np
import h5py
import weakref
import imageio
import random
from utils.dsec_utils import flow_16bit_to_float, EventSlicer, VoxelGrid
import argparse
import tqdm

'''
DSEC Dataset
└── Train
    ├── train_events
    │   ├── thun_00_a
    │   │   ├── events
    │   │   │   ├── left
    │   │   │   │   ├── events.h5
    │   │   │   │   └── rectify_map.h5
    │   │   │   │   └── rect2dist_map.npy
    ├── train_optical_flow
    │   ├── thun_00_a
    │   │   ├── flow
    │   │   │   ├── forward
    │   │   │   │   ├── xxxxxx.png
    │   │   │   ├── forward_timestamps.txt
'''

class VoxelGridSequenceDSEC(Dataset):
    def __init__(self, events_sequence_path: Path, flow_sequence_path: Path,
                 crop_size=None):
        assert events_sequence_path.is_dir()
        assert flow_sequence_path.is_dir()
        '''
        Directory Structure:

        Dataset
        └── Train
            ├── train_events
            │   ├── thun_00_a (events_sequence_path)
            │   │   ├── events
            │   │   │   ├── left
            │   │   │   │   ├── events.h5
            │   │   │   │   └── rectify_map.h5
            ├── train_optical_flow
            │   ├── thun_00_a (flow_sequence_path)
            │   │   ├── flow
            │   │   │   ├── forward
            │   │   │   │   ├── xxxxxx.png
            │   │   │   ├── forward_timestamps.txt
        '''
        forward_timestamps_file = flow_sequence_path / 'flow' / 'forward_timestamps.txt'
        assert forward_timestamps_file.is_file()
        self.forward_ts_pair = np.genfromtxt(forward_timestamps_file, delimiter=',')
        self.forward_ts_pair = self.forward_ts_pair.astype('int64')
        '''
        ----- forward_timestamps.txt -----
        # from_timestamp_us, to_timestamp_us
        49599300523, 49599400524
        49599400524, 49599500511
        49599500511, 49599600529
        49599600529, 49599700535
        49599700535, 49599800517
        ...
        '''
        self.height = 480
        self.width = 640
        self.voxel_grid = VoxelGrid((15, self.height, self.width), normalize=True)
        # 光流的时间间隔
        self.delta_t_us = 100000

        ev_dir = events_sequence_path / 'events' / 'left'
        ev_data_file = ev_dir / 'events.h5'
        ev_rect_file = ev_dir / 'rectify_map.h5'

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_map = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        flow_images_path = flow_sequence_path / 'flow' / 'forward'
        flow_images_files = [x for x in flow_images_path.iterdir()]
        self.gt_flow_files = sorted(flow_images_files, key=lambda x: int(x.stem))

        # self.transform = transforms
        self.crop_size = crop_size
        if self.crop_size is not None:
            assert self.crop_size[0] <= self.crop_size[1]

    def events_to_voxel_grid(self, p, t, x, y, device: str = 'cpu'):
        t = (t - t[0]).astype('float32')
        t = (t / t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.forward_ts_pair)

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def get_data_sample(self, index, crop_window=None):
        '''
        output.keys = ['event_volume_old', 'event_volume_new', 'flow', 'valid2D']
        '''
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        names = ['event_volume_old', 'event_volume_new']
        ts_begin_i = self.forward_ts_pair[index, 0]
        ts_start = [ts_begin_i - self.delta_t_us, ts_begin_i]
        ts_end = [ts_begin_i, ts_begin_i + self.delta_t_us]
        output = {}
        flow, valid2D = VoxelGridSequenceDSEC.load_flow(self.gt_flow_files[index])
        output['flow'] = torch.from_numpy(flow)
        # print('flow-size', output['flow'].shape)
        output['valid2D'] = torch.from_numpy(valid2D)
        # print('valid-size', output['valid2D'].shape)
        # 获取事件的 voxel_gird
        for i in range(len(names)):
            event_data = self.event_slicer.get_events(ts_start[i], ts_end[i])

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            # 得到畸变矫正之后的 x_rect 和 y_rect
            xy_rect = self.rectify_events(x, y)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            # 窗口裁剪，去掉一些事件，并非数据增强中的裁剪
            if crop_window is not None:
                # Cropping (+- 2 for safety reasons)
                x_mask = (x_rect >= crop_window['start_x'] - 2) & (
                        x_rect < crop_window['start_x'] + crop_window['crop_width'] + 2)
                y_mask = (y_rect >= crop_window['start_y'] - 2) & (
                        y_rect < crop_window['start_y'] + crop_window['crop_height'] + 2)
                mask_combined = x_mask & y_mask
                p = p[mask_combined]
                t = t[mask_combined]
                x_rect = x_rect[mask_combined]
                y_rect = y_rect[mask_combined]

            if self.voxel_grid is None:
                raise NotImplementedError
            else:
                event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
                output[names[i]] = event_representation

        # random crop and random flip
        output['flow'] = output['flow'].permute(2, 0, 1)
        if self.crop_size is not None:
            # get rand h_start and w_start
            rand_max_h = self.height - self.crop_size[0]
            rand_max_w = self.width - self.crop_size[1]
            h_start = random.randrange(0, rand_max_h + 1)
            w_start = random.randrange(0, rand_max_w + 1)
            # random flip transform
            p = random.randint(0, 1)
            flip_param = 1 - 2 * p
            flip = transforms.RandomHorizontalFlip(p)
            # apply transform
            for key in output.keys():
                # print(key, output[key].shape)
                output[key] = output[key][..., h_start:h_start + self.crop_size[0], w_start:w_start + self.crop_size[1]]
                output[key] = flip(output[key])
            # flip the flow value of u
            output['flow'][0] = flip_param * output['flow'][0]

        return output['event_volume_old'], output['event_volume_new'], output['flow'], output['valid2D'].float()

    def __getitem__(self, idx):
        return self.get_data_sample(idx)


class VoxelGridSequenceDSECTest(Dataset):
    def __init__(self, events_sequence_path: Path, forward_ts_file: Path, bins=15):
        assert events_sequence_path.is_dir()
        assert forward_ts_file.is_file()
        '''
        Directory Structure:

        Dataset
        └── Train
            ├── train_events
            │   ├── thun_00_a (events_sequence_path)
            │   │   ├── events
            │   │   │   ├── left
            │   │   │   │   ├── events.h5
            │   │   │   │   └── rectify_map.h5
            ├── train_optical_flow
            │   ├── thun_00_a (flow_sequence_path)
            │   │   ├── flow
            │   │   │   ├── forward
            │   │   │   │   ├── xxxxxx.png
            │   │   │   ├── forward_timestamps.txt
        '''
        self.forward_ts_pair_idx = np.genfromtxt(forward_ts_file, delimiter=',')
        self.forward_ts_pair_idx = self.forward_ts_pair_idx.astype('int64')
        '''
        ----- forward_timestamps.txt -----
        # from_timestamp_us, to_timestamp_us
        49599300523, 49599400524
        49599400524, 49599500511
        49599500511, 49599600529
        49599600529, 49599700535
        49599700535, 49599800517
        ...
        '''
        self.height = 480
        self.width = 640
        self.bins = bins
        self.voxel_grid = VoxelGrid((bins, self.height, self.width), normalize=True)
        # 光流的时间间隔
        self.delta_t_us = 100000

        ev_dir = events_sequence_path / 'events' / 'left'
        ev_data_file = ev_dir / 'events.h5'
        ev_rect_file = ev_dir / 'rectify_map.h5'

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_map = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)


    def events_to_voxel_grid(self, p, t, x, y, device: str = 'cpu'):
        t = (t - t[0]).astype('float32')
        t = (t / t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        event_data_torch = {
            'p': torch.from_numpy(pol),
            't': torch.from_numpy(t),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
        }
        return self.voxel_grid.convert(event_data_torch)

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.forward_ts_pair_idx)

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def get_data_sample(self, index, crop_window=None):
        '''
        output.keys = ['event_volume_old', 'event_volume_new', 'flow', 'valid2D']
        '''
        # First entry corresponds to all events BEFORE the flow map
        # Second entry corresponds to all events AFTER the flow map (corresponding to the actual fwd flow)
        names = ['event_volume_old', 'event_volume_new']
        ts_begin_i = self.forward_ts_pair_idx[index, 0]
        output_name_idx = self.forward_ts_pair_idx[index, 2]
        ts_start = [ts_begin_i - self.delta_t_us, ts_begin_i]
        ts_end = [ts_begin_i, ts_begin_i + self.delta_t_us]
        output = {}
        # 获取事件的 voxel_gird
        for i in range(len(names)):
            event_data = self.event_slicer.get_events(ts_start[i], ts_end[i])

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            # 得到畸变矫正之后的 x_rect 和 y_rect
            xy_rect = self.rectify_events(x, y)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            # 窗口裁剪，去掉一些事件，并非数据增强中的裁剪
            if crop_window is not None:
                # Cropping (+- 2 for safety reasons)
                x_mask = (x_rect >= crop_window['start_x'] - 2) & (
                        x_rect < crop_window['start_x'] + crop_window['crop_width'] + 2)
                y_mask = (y_rect >= crop_window['start_y'] - 2) & (
                        y_rect < crop_window['start_y'] + crop_window['crop_height'] + 2)
                mask_combined = x_mask & y_mask
                p = p[mask_combined]
                t = t[mask_combined]
                x_rect = x_rect[mask_combined]
                y_rect = y_rect[mask_combined]

            if self.voxel_grid is None:
                raise NotImplementedError
            else:
                event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
                output[names[i]] = event_representation

        return output['event_volume_old'], output['event_volume_new'], output_name_idx

    def __getitem__(self, idx):
        return self.get_data_sample(idx)


class VoxelGridDatasetProviderDSEC:
    def __init__(self, dataset_path: Path, crop_size=None, random_split_seed: int = 42, train_ratio=0.8):
        events_sequence_path = dataset_path / 'Train' / 'train_events'
        flow_sequence_path = dataset_path / 'Train' / 'train_optical_flow'
        events_sequences = set([x.stem for x in events_sequence_path.iterdir()])
        flow_sequences = set([x.stem for x in flow_sequence_path.iterdir()])
        valid_sequences = events_sequences.intersection(flow_sequences)
        valid_sequences = list(valid_sequences)
        valid_sequences.sort()

        dataset_sequences = list()
        dataset_sequences_cropped = list()
        for sequence in valid_sequences:
            dataset_sequences.append(VoxelGridSequenceDSEC(events_sequence_path / sequence,
                                                           flow_sequence_path / sequence,
                                                           crop_size=None))
            dataset_sequences_cropped.append(VoxelGridSequenceDSEC(events_sequence_path / sequence,
                                                                   flow_sequence_path / sequence,
                                                                   crop_size=crop_size))

        self.dataset = torch.utils.data.ConcatDataset(dataset_sequences)
        self.dataset_cropped = torch.utils.data.ConcatDataset(dataset_sequences_cropped)
        self.full_size = len(self.dataset)
        self.train_size = int(train_ratio * self.full_size)
        self.valid_size = self.full_size - self.train_size

        generator = torch.Generator().manual_seed(random_split_seed)
        lengths = [self.train_size, self.valid_size]
        indices = torch.randperm(sum(lengths), generator=generator).tolist()

        self.train_indices = indices[0: self.train_size]
        self.train_indices.sort()

        self.valid_indices = indices[self.train_size:self.full_size]
        self.valid_indices.sort()
        self.train_set = Subset(self.dataset, self.train_indices)
        self.valid_set = Subset(self.dataset, self.valid_indices)
        self.train_set_cropped = Subset(self.dataset_cropped, self.train_indices)
        self.valid_set_cropped = Subset(self.dataset_cropped, self.valid_indices)


class FastVoxelGridSequenceDSEC(Dataset):
    def __init__(self, events_sequence_path: Path, flow_sequence_path: Path,
                 bins=15, crop_size=None, return_raw=False, unified=True, norm=True):
        assert events_sequence_path.is_dir()
        assert flow_sequence_path.is_dir()
        '''
        Directory Structure:

        Dataset
        └── Train
            ├── train_events
            │   ├── thun_00_a (events_sequence_path)
            │   │   ├── events
            │   │   │   ├── left
            │   │   │   │   ├── events.h5
            │   │   │   │   └── rectify_map.h5
            ├── train_optical_flow
            │   ├── thun_00_a (flow_sequence_path)
            │   │   ├── flow
            │   │   │   ├── forward
            │   │   │   │   ├── xxxxxx.png
            │   │   │   ├── forward_timestamps.txt
        '''
        forward_timestamps_file = flow_sequence_path / 'flow' / 'forward_timestamps.txt'
        assert forward_timestamps_file.is_file()
        self.forward_ts_pair = np.genfromtxt(forward_timestamps_file, delimiter=',')
        self.forward_ts_pair = self.forward_ts_pair.astype('int64')
        self.bins = bins
        self.return_raw = return_raw
        self.unified = unified
        self.norm = norm
        '''
        ----- forward_timestamps.txt -----
        # from_timestamp_us, to_timestamp_us
        49599300523, 49599400524
        49599400524, 49599500511
        49599500511, 49599600529
        49599600529, 49599700535
        49599700535, 49599800517
        ...
        '''
        self.height = 480
        self.width = 640
        # 光流的时间间隔
        self.delta_t_us = 100000

        ev_dir = events_sequence_path / 'events' / 'left'
        ev_data_file = ev_dir / 'events.h5'
        ev_rect_file = ev_dir / 'rectify_map.h5'
        rect2dist_map_path = ev_dir / 'rect2dist_map.npy'
        self.rect2dist_map = torch.from_numpy(np.load(str(rect2dist_map_path)))
        rect2dist_map = self.rect2dist_map.clone()
        rect2dist_map[..., 0] = 2 * rect2dist_map[..., 0] / (self.width - 1) - 1
        rect2dist_map[..., 1] = 2 * rect2dist_map[..., 1] / (self.height - 1) - 1
        self.grid = rect2dist_map.unsqueeze(0)

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_map = h5_rect['rectify_map'][()]

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        flow_images_path = flow_sequence_path / 'flow' / 'forward'
        flow_images_files = [x for x in flow_images_path.iterdir()]
        self.gt_flow_files = sorted(flow_images_files, key=lambda x: int(x.stem))

        # self.transform = transforms
        self.crop_size = crop_size
        if self.crop_size is not None:
            assert self.crop_size[0] <= self.crop_size[1]

    def events_to_unified_voxel_grid(self, p, t, x, y, bins):
        t_norm = torch.from_numpy((t - t[0]).astype('float32'))
        # t_norm = (self.bins - 1) * (t_norm / t_norm[-1])
        bin_time = self.delta_t_us / (self.bins - 1)
        total_t = 2 * bin_time + self.delta_t_us
        assert total_t > t_norm[-1]
        t_norm = (self.bins - 1 + 2) * (t_norm / total_t)
        x = torch.from_numpy(x.astype('float32')).int()
        y = torch.from_numpy(y.astype('float32')).int()
        # convert p (0, 1) → (-1, 1)
        p_value = 2 * torch.from_numpy(p.astype('float32')) - 1

        t0 = t_norm.int()
        H, W = self.height, self.width
        fast_voxel_grid = torch.zeros((bins + 2, H, W), dtype=torch.float, requires_grad=False)
        for tlim in [t0, t0 + 1]:
            mask = tlim < self.bins + 2
            index = H * W * tlim.long() + \
                    W * y.long() + \
                    x.long()
            interp_weights = p_value * (1 - (t_norm - tlim).abs())
            fast_voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return fast_voxel_grid[1:-1, :, :]

    def events_to_voxel_grid(self, p, t, x, y, bins):
        t_norm = torch.from_numpy((t - t[0]).astype('float32'))
        # t_norm = (self.bins - 1) * (t_norm / t_norm[-1])
        assert self.delta_t_us > t_norm[-1]
        t_norm = (self.bins - 1) * (t_norm / self.delta_t_us)
        x = torch.from_numpy(x.astype('float32')).int()
        y = torch.from_numpy(y.astype('float32')).int()
        # convert p (0, 1) → (-1, 1)
        p_value = 2 * torch.from_numpy(p.astype('float32')) - 1

        t0 = t_norm.int()
        H, W = self.height, self.width
        fast_voxel_grid = torch.zeros((bins, H, W), dtype=torch.float, requires_grad=False)
        for tlim in [t0, t0 + 1]:
            mask = tlim < self.bins
            index = H * W * tlim.long() + \
                    W * y.long() + \
                    x.long()
            interp_weights = p_value * (1 - (t_norm - tlim).abs())
            fast_voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return fast_voxel_grid

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.forward_ts_pair)

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def warp_rectify(self, voxel_gird):
        rect_grid = F.grid_sample(voxel_gird.unsqueeze(0), self.grid, align_corners=True)
        return rect_grid.squeeze(0)

    def get_data_sample(self, index):
        '''
        output.keys = ['events_vg', 'flow', 'valid2D']
        '''
        ts_begin_i = self.forward_ts_pair[index, 0]
        bin_time = self.delta_t_us / (self.bins - 1)
        ts_start_extend = ts_begin_i - bin_time
        ts_end_extend = ts_begin_i + self.delta_t_us + bin_time
        output = {}
        flow, valid2D = VoxelGridSequenceDSEC.load_flow(self.gt_flow_files[index])
        output['flow'] = torch.from_numpy(flow)
        # print('flow-size', output['flow'].shape)
        output['valid2D'] = torch.from_numpy(valid2D)
        # print('valid-size', output['valid2D'].shape)
        raw_events = None

        if self.crop_size is not None:
            # get rand h_start and w_start
            rand_max_h = self.height - self.crop_size[0]
            rand_max_w = self.width - self.crop_size[1]
            h_start = random.randrange(0, rand_max_h + 1)
            w_start = random.randrange(0, rand_max_w + 1)
            W = self.crop_size[1]
            H = self.crop_size[0]
        else:
            h_start = 0
            w_start = 0
            W = self.width
            H = self.height

        is_unified = self.unified
        event_data = None
        if self.unified:
            # 获取事件的 voxel_gird_extend
            event_data = self.event_slicer.get_events(ts_start_extend, ts_end_extend)
        if (event_data is None) or (not self.unified):
            """
            if time extend the scope, return voxel_grid rather than voxel_grid_extend
            """
            event_data = self.event_slicer.get_events(ts_begin_i, ts_begin_i + self.delta_t_us)
            is_unified = False

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']
        xy_rect = self.rectify_events(event_data['x'], event_data['y'])
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        x_mask = (x_rect >= w_start - 2) & (
                x_rect < w_start + W + 2)
        y_mask = (y_rect >= h_start - 2) & (
                y_rect < h_start + H + 2)
        mask_combined = x_mask & y_mask
        x = x[mask_combined]
        y = y[mask_combined]
        p = p[mask_combined]
        t = t[mask_combined]
        x_rect = x_rect[mask_combined]
        y_rect = y_rect[mask_combined]

        if is_unified:
            fast_voxel_grid = self.events_to_unified_voxel_grid(p, t, x, y, self.bins)
            mask_time = (t >= ts_begin_i) & (t < ts_begin_i + self.delta_t_us)
            x_rect = x_rect[mask_time]
            y_rect = y_rect[mask_time]
            p = p[mask_time]
            t = t[mask_time]
        else:
            fast_voxel_grid = self.events_to_voxel_grid(p, t, x, y, self.bins)

        event_len = t.size
        if self.return_raw and event_len != 0:
            raw_events = torch.zeros(event_len, 4)
            raw_events[:, 0] = torch.from_numpy(x_rect)
            raw_events[:, 1] = torch.from_numpy(y_rect)
            raw_events[:, 3] = 2 * torch.from_numpy(p.astype('float32')) - 1
            raw_events[:, 2] = torch.from_numpy(
                (t - ts_begin_i).astype('float32')) / self.delta_t_us * (self.bins - 1)


        # warp to rectify distortion
        output['events_vg'] = self.warp_rectify(fast_voxel_grid)

        p_flip = 0
        # random crop and random flip
        output['flow'] = output['flow'].permute(2, 0, 1)
        if self.crop_size is not None:
            # random flip transform
            p_flip = random.randint(0, 1)
            flip_param = 1 - 2 * p_flip
            flip = transforms.RandomHorizontalFlip(p_flip)
            # apply transform
            for key in output.keys():
                # print(key, output[key].shape)
                output[key] = output[key][..., h_start:h_start + self.crop_size[0], w_start:w_start + self.crop_size[1]]
                output[key] = flip(output[key])
            # flip the flow value of u
            output['flow'][0] = flip_param * output['flow'][0]

        if self.return_raw and (raw_events is not None) and raw_events.size()[0] != 0:
            crop_mask = (raw_events[:, 0] >= w_start) & (raw_events[:, 0] < w_start + W-1)
            crop_mask &= (raw_events[:, 1] >= h_start) & (raw_events[:, 1] < h_start + H-1)
            if torch.any(crop_mask):
                raw_events = raw_events[crop_mask]
                raw_events[:, 0] -= w_start
                raw_events[:, 1] -= h_start
                if p_flip == 1:
                    raw_events[:, 0] = W - 1 - raw_events[:, 0]
                # exchange [x, y, t, p] to [y, x, t, p]
                raw_events = raw_events[:, [1, 0, 2, 3]]
            else:
                raw_events = None
        else:
            raw_events = None

        if self.norm:
            mask = torch.nonzero(output['events_vg'], as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = output['events_vg'][mask].mean()
                std = output['events_vg'][mask].std()
                if std > 0:
                    output['events_vg'][mask] = (output['events_vg'][mask] - mean) / std
                else:
                    output['events_vg'][mask] = output['events_vg'][mask] - mean

        # raw_events的维度信息和含义:
        # 维度: [N, 4], 其中N是事件的数量
        # 每一行的4个值分别表示:
        # [0]: y 坐标 (高度方向)
        # [1]: x 坐标 (宽度方向) 
        # [2]: 归一化的时间戳 (范围0到bins-1)
        # [3]: 极性 (-1或1)
        if self.return_raw:
            return output['events_vg'], output['flow'], output['valid2D'].float(), raw_events
        return output['events_vg'], output['flow'], output['valid2D'].float()

    def __getitem__(self, idx):
        return self.get_data_sample(idx)

class FastVoxelGridDatasetProviderDSEC:
    def __init__(self, dataset_path: Path, bins=15, crop_size=[288, 384],
                 random_split_seed: int = 42, train_ratio=0.8, return_raw=False, unified=True, norm=True):
        events_sequence_path = dataset_path / 'Train' / 'train_events'
        flow_sequence_path = dataset_path / 'Train' / 'train_optical_flow'
        events_sequences = set([x.stem for x in events_sequence_path.iterdir()])
        flow_sequences = set([x.stem for x in flow_sequence_path.iterdir()])
        valid_sequences = events_sequences.intersection(flow_sequences)
        valid_sequences = list(valid_sequences)
        valid_sequences.sort()

        dataset_sequences = list()
        dataset_sequences_cropped = list()
        for sequence in valid_sequences:
            dataset_sequences.append(FastVoxelGridSequenceDSEC(events_sequence_path / sequence,
                                                             flow_sequence_path / sequence,
                                                             bins=bins,
                                                             crop_size=None,
                                                             return_raw=return_raw,
                                                             unified=unified,
                                                             norm=norm))
            dataset_sequences_cropped.append(FastVoxelGridSequenceDSEC(events_sequence_path / sequence,
                                                                     flow_sequence_path / sequence,
                                                                     bins=bins,
                                                                     crop_size=crop_size,
                                                                     return_raw=return_raw,
                                                                     unified=unified,
                                                                     norm=norm))

        self.dataset = torch.utils.data.ConcatDataset(dataset_sequences)
        self.dataset_cropped = torch.utils.data.ConcatDataset(dataset_sequences_cropped)
        self.full_size = len(self.dataset)
        self.train_size = int(train_ratio * self.full_size)
        self.valid_size = self.full_size - self.train_size

        generator = torch.Generator().manual_seed(random_split_seed)
        lengths = [self.train_size, self.valid_size]
        indices = torch.randperm(sum(lengths), generator=generator).tolist()

        self.train_indices = indices[0: self.train_size]
        self.train_indices.sort()

        self.valid_indices = indices[self.train_size:self.full_size]
        self.valid_indices.sort()
        self.train_set = Subset(self.dataset, self.train_indices)
        self.valid_set = Subset(self.dataset, self.valid_indices)
        self.train_set_cropped = Subset(self.dataset_cropped, self.train_indices)
        self.valid_set_cropped = Subset(self.dataset_cropped, self.valid_indices)

class FastVoxelGridTestSequenceDSEC(Dataset):
    def __init__(self, events_sequence_path: Path, forward_ts_file: Path, bins=15, return_raw=False, unified=True, norm=True):
        """
        get a sequence
        :param events_sequence_path: Path, events_sequence_path
        :param flow_sequence_path: Path, flow_sequence_path
        :param raster_channels: compressed events channels
        """
        assert events_sequence_path.is_dir()
        assert forward_ts_file.is_file()
        '''
        Directory Structure:

        Dataset
        └── Train
            ├── train_events
            │   ├── thun_00_a (events_sequence_path)
            │   │   ├── events
            │   │   │   ├── left
            │   │   │   │   ├── events.h5
            │   │   │   │   └── rectify_map.h5
            │   │   │   │   └── rect2dist_map.npy
        '''
        self.forward_ts_pair_idx = np.genfromtxt(forward_ts_file, delimiter=',')
        self.forward_ts_pair_idx = self.forward_ts_pair_idx.astype('int64')
        '''
        ----- forward_ts_file -----
        # from_timestamp_us, to_timestamp_us, file_index
        51648500652, 51648600574, 820
        51649000383, 51649100410, 830
        51649500439, 51649600452, 840
        51650000446, 51650100510, 850
        51650500682, 51650600786, 860
        51651002403, 51651103123, 870
        51651507548, 51651607608, 880
        51652007591, 51652107617, 890
        51652507627, 51652607642, 900
        ...
        '''
        self.height = 480
        self.width = 640
        self.bins = bins
        # 光流的时间间隔 100000 us
        self.delta_t_us = 100000
        self.return_raw = return_raw
        self.unified = unified
        self.norm = norm

        ev_dir = events_sequence_path / 'events' / 'left'
        ev_data_file = ev_dir / 'events.h5'
        ev_rect_file = ev_dir / 'rectify_map.h5'

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_map = h5_rect['rectify_map'][()]
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        rect2dist_map_path = ev_dir / 'rect2dist_map.npy'
        self.rect2dist_map = torch.from_numpy(np.load(str(rect2dist_map_path)))
        rect2dist_map = self.rect2dist_map.clone()
        rect2dist_map[..., 0] = 2 * rect2dist_map[..., 0] / (self.width - 1) - 1
        rect2dist_map[..., 1] = 2 * rect2dist_map[..., 1] / (self.height - 1) - 1
        self.grid = rect2dist_map.unsqueeze(0)

    def events_to_unified_voxel_grid(self, p, t, x, y, bins):
        t_norm = torch.from_numpy((t - t[0]).astype('float32'))
        # t_norm = (self.bins - 1) * (t_norm / t_norm[-1])
        bin_time = self.delta_t_us / (self.bins - 1)
        total_t = 2 * bin_time + self.delta_t_us
        assert total_t > t_norm[-1]
        t_norm = (self.bins - 1 + 2) * (t_norm / total_t)
        x = torch.from_numpy(x.astype('float32')).int()
        y = torch.from_numpy(y.astype('float32')).int()
        # convert p (0, 1) → (-1, 1)
        p_value = 2 * torch.from_numpy(p.astype('float32')) - 1

        t0 = t_norm.int()
        H, W = self.height, self.width
        fast_voxel_grid = torch.zeros((bins + 2, H, W), dtype=torch.float, requires_grad=False)
        for tlim in [t0, t0 + 1]:
            mask = tlim < self.bins + 2
            index = H * W * tlim.long() + \
                    W * y.long() + \
                    x.long()
            interp_weights = p_value * (1 - (t_norm - tlim).abs())
            fast_voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return fast_voxel_grid[1:-1, :, :]

    def events_to_voxel_grid(self, p, t, x, y, bins):
        t_norm = torch.from_numpy((t - t[0]).astype('float32'))
        # t_norm = (self.bins - 1) * (t_norm / t_norm[-1])
        assert self.delta_t_us > t_norm[-1]
        t_norm = (self.bins - 1) * (t_norm / self.delta_t_us)
        x = torch.from_numpy(x.astype('float32')).int()
        y = torch.from_numpy(y.astype('float32')).int()
        # convert p (0, 1) → (-1, 1)
        p_value = 2 * torch.from_numpy(p.astype('float32')) - 1

        t0 = t_norm.int()
        H, W = self.height, self.width
        fast_voxel_grid = torch.zeros((bins, H, W), dtype=torch.float, requires_grad=False)
        for tlim in [t0, t0 + 1]:
            mask = tlim < self.bins
            index = H * W * tlim.long() + \
                    W * y.long() + \
                    x.long()
            interp_weights = p_value * (1 - (t_norm - tlim).abs())
            fast_voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return fast_voxel_grid

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.forward_ts_pair_idx)

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_map
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def warp_rectify(self, voxel_gird):
        rect_grid = F.grid_sample(voxel_gird.unsqueeze(0), self.grid, align_corners=True)
        return rect_grid.squeeze(0)

    def get_data_sample(self, index):
        '''
        output.keys = ['events_vg', 'name']
        '''
        ts_begin_i = self.forward_ts_pair_idx[index, 0]
        output_name_idx = self.forward_ts_pair_idx[index, 2]
        ts_end_i = self.forward_ts_pair_idx[index, 1]
        bin_time = self.delta_t_us / (self.bins - 1)
        ts_start_extend = ts_begin_i - bin_time
        ts_end_extend = ts_begin_i + self.delta_t_us + bin_time
        output = {}
        raw_events = None

        h_start = 0
        w_start = 0
        W = self.width
        H = self.height

        is_unified = self.unified
        event_data = None
        if self.unified:
            # 获取事件的 voxel_gird_extend
            event_data = self.event_slicer.get_events(ts_start_extend, ts_end_extend)
        if (event_data is None) or (not self.unified):
            """
            if time extend the scope, return voxel_grid rather than voxel_grid_extend
            """
            event_data = self.event_slicer.get_events(ts_begin_i, ts_begin_i + self.delta_t_us)
            is_unified = False

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']
        xy_rect = self.rectify_events(event_data['x'], event_data['y'])
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        x_mask = (x_rect >= w_start - 2) & (
                x_rect < w_start + W + 2)
        y_mask = (y_rect >= h_start - 2) & (
                y_rect < h_start + H + 2)
        mask_combined = x_mask & y_mask
        x = x[mask_combined]
        y = y[mask_combined]
        p = p[mask_combined]
        t = t[mask_combined]
        x_rect = x_rect[mask_combined]
        y_rect = y_rect[mask_combined]

        if is_unified:
            fast_voxel_grid = self.events_to_unified_voxel_grid(p, t, x, y, self.bins)
            mask_time = (t >= ts_begin_i) & (t < ts_begin_i + self.delta_t_us)
            x_rect = x_rect[mask_time]
            y_rect = y_rect[mask_time]
            p = p[mask_time]
            t = t[mask_time]
        else:
            fast_voxel_grid = self.events_to_voxel_grid(p, t, x, y, self.bins)

        # warp to rectify distortion
        out_voxel_grid = self.warp_rectify(fast_voxel_grid)

        event_len = t.size
        if self.return_raw and event_len != 0:
            raw_events = torch.zeros(event_len, 4)
            raw_events[:, 0] = torch.from_numpy(x_rect)
            raw_events[:, 1] = torch.from_numpy(y_rect)
            raw_events[:, 3] = 2 * torch.from_numpy(p.astype('float32')) - 1
            raw_events[:, 2] = torch.from_numpy(
                (t - ts_begin_i).astype('float32')) / self.delta_t_us * (self.bins - 1)

        if self.return_raw and (raw_events is not None) and raw_events.size()[0] != 0:
            crop_mask = (raw_events[:, 0] >= w_start) & (raw_events[:, 0] < w_start + W-1)
            crop_mask &= (raw_events[:, 1] >= h_start) & (raw_events[:, 1] < h_start + H-1)
            if torch.any(crop_mask):
                raw_events = raw_events[crop_mask]
                raw_events[:, 0] -= w_start
                raw_events[:, 1] -= h_start
                # exchange [x, y, t, p] to [y, x, t, p] for event_image_converter
                raw_events = raw_events[:, [1, 0, 2, 3]]
            else:
                raw_events = None
        else:
            raw_events = None

        if self.norm:
            mask = torch.nonzero(out_voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = out_voxel_grid[mask].mean()
                std = out_voxel_grid[mask].std()
                if std > 0:
                    out_voxel_grid[mask] = (out_voxel_grid[mask] - mean) / std
                else:
                    out_voxel_grid[mask] = out_voxel_grid[mask] - mean

        if self.return_raw:
            return out_voxel_grid, output_name_idx, raw_events
        return out_voxel_grid, output_name_idx

    def __getitem__(self, idx):
        return self.get_data_sample(idx)


def collate_raw_events(data):
    from torch.utils.data.dataloader import default_collate
    voxel_grid_list = list()
    flow_list = list()
    mask_list = list()
    raw_events_list = list()
    # 遍历数据列表中的每个元素
    for i, d in enumerate(data):
        voxel_grid_list.append(d[0])  # 将体素网格数据添加到列表中
        flow_list.append(d[1])  # 将光流数据添加到列表中
        mask_list.append(d[2])  # 将掩码数据添加到列表中
        if d[3] is not None:
            # 如果原始事件数据存在，将批次索引与事件数据拼接后添加到列表中
            raw_events_list.append(torch.cat([i*torch.ones(len(d[3]), 1), d[3]], 1))

    # 使用default_collate函数将列表中的数据合并成张量
    out_voxel_grid = default_collate(voxel_grid_list)
    out_flow = default_collate(flow_list)
    out_mask = default_collate(mask_list)
    
    out_raw_events = None
    if len(raw_events_list)!=0:
        # 如果存在原始事件数据，将所有批次的事件数据拼接成一个大张量
        out_raw_events = torch.cat(raw_events_list, dim=0)
        # out_raw_events的维度信息和含义:
        # 维度: [N, 5], 其中N是所有批次中事件的总数量
        # 每一行的5个值分别表示:
        # [0]: 批次索引
        # [1]: y 坐标 (高度方向)
        # [2]: x 坐标 (宽度方向) 
        # [3]: 归一化的时间戳 (范围0到bins-1)
        # [4]: 极性 (-1或1)
    
    # 返回处理后的数据
    return out_voxel_grid, out_flow, out_mask, out_raw_events


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/media/yyz/FastDisk/Dataset/DSEC")
    parser.add_argument('--bins', type=int, default=15)
    parser.add_argument('--crop_size', type=int, default=[288, 384])
    parser.add_argument('--random_split_seed', type=int, default=42)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--return_raw', type=bool, default=True)
    parser.add_argument('--unified', type=bool, default=True)
    parser.add_argument('--norm', type=bool, default=True)
    args = parser.parse_args()

    import utils.setupTensor

    dsec_provider = FastVoxelGridDatasetProviderDSEC(Path(args.dataset_path), bins=args.bins,
                                                   crop_size=args.crop_size,
                                                   random_split_seed=args.random_split_seed,
                                                   train_ratio=args.train_ratio,
                                                   return_raw=args.return_raw,
                                                   unified=args.unified,
                                                   norm=args.norm)
    
    dataset = dsec_provider.train_set_cropped
    loader = DataLoader(dataset, batch_size=3, num_workers=4, drop_last=False, shuffle=True, collate_fn=collate_raw_events)
    for i, data_blob in enumerate(tqdm.tqdm(loader)):
        voxel_grid, flow, valid_mask, events = data_blob
        voxel_grid, flow, valid_mask = [x.cuda() for x in [voxel_grid, flow, valid_mask]]
        if events is not None:
            events = events.cuda()

