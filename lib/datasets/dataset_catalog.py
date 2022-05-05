class DatasetCatalog(object):
    dataset_attrs = {
        'SynTrain': {
            'data_root': 'data/nerf_synthetic',
            'ann_file': 'data/cache/mvsnerf/pairs.th',
            'split': 'train'
        },
        'SynVal': {
            'data_root': 'data/nerf_synthetic',
            'ann_file': 'data/cache/mvsnerf/pairs.th',
            'split': 'val'
        },
        'LSTrain': {
            'data_root': 'data/CoreView_313',
            'ann_file': '',
            'split': 'train'
        },
        'LSVal': {
            'data_root': 'data/CoreView_313',
            'ann_file': '',
            'split': 'val'
        },
        'LLFFTrain': {
            'data_root': 'data/nerf_llff_data',
            'ann_file': '',
            'split': 'train'
        },
        'LLFFVal': {
            'data_root': 'data/nerf_llff_data',
            'ann_file': '',
            'split': 'val'
        },
        'DtuTrain': {
            'data_root': 'data/dtu',
            'ann_file': 'data/cache/mvsnerf/dtu_train_all.txt',
            'split': 'train'
        },
        'DtuminiVal': {
            'data_root': 'data/dtu',
            'ann_file': 'data/cache/mvsnerf/dtu_minival.txt',
            'split': 'val'
        },
        'DtuVal': {
            'data_root': 'data/dtu',
            'ann_file': ['data/cache/mvsnerf/dtu_val_all.txt',
                'data/cache/mvsnerf/pairs.th'],
            'split': 'val'
        },
        'DtuftVal': {
            'data_root': 'data/dtu',
            'ann_file': 'data/cache/mvsnerf/pairs.th',
            'split': 'val'
        },
        'DtuftTrain': {
            'data_root': 'data/dtu',
            'ann_file':  'data/cache/mvsnerf/pairs.th',
            'split': 'train'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
