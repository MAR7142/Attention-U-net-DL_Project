import argparse

from torch.utils.data import DataLoader

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.error_logger import ErrorLogger

from models import get_model


def evaluate(config_path, which_epoch):
    # Load options
    json_opts = json_file_to_pyobj(config_path)
    train_opts = json_opts.training
    arch_type = train_opts.arch_type

    # Force the checkpoint-loading branch in FeedForwardSegmentation.initialize()
    # (isTrain=True so self.criterion gets built, which model.validate() needs;
    # continue_train=True + which_epoch=N so '{which_epoch:03d}_net_S.pth' is loaded
    # from the config's checkpoints_dir/experiment_name), regardless of what the
    # config file itself has these fields set to.
    model_overrides = {'isTrain': True, 'continue_train': True, 'which_epoch': which_epoch}
    if 'path_pre_trained_model' in json_opts.model._fields:
        # Make sure the which_epoch-based checkpoint path is used, not an explicit path.
        model_overrides['path_pre_trained_model'] = None
    model_opts = json_opts.model._replace(**model_overrides)

    # Setup the NN Model - this loads the requested checkpoint
    model = get_model(model_opts)

    # Setup Dataset and Augmentation - test split only, using the same deterministic
    # 'valid' transform (no random flip/affine) that train_segmentation.py uses for test.
    ds_class = get_dataset(arch_type)
    ds_path = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    test_dataset = ds_class(ds_path, split='test', transform=ds_transform['valid'],
                             preload_data=train_opts.preloadData)
    test_loader = DataLoader(dataset=test_dataset, num_workers=0,
                              batch_size=train_opts.batchSize, shuffle=False)

    # Same stats path already used and verified in train_segmentation.py's validation/
    # test loop: model.validate() -> model.get_segmentation_stats() -> ErrorLogger.
    # This is n_class-agnostic (driven by opts.output_nc), unlike validation.py's
    # dice_score(..., n_class=4), which is hardcoded for the 4-class cardiac task.
    error_logger = ErrorLogger()
    for images, labels in test_loader:
        model.set_input(images, labels)
        model.validate()
        stats = model.get_segmentation_stats()
        error_logger.update(stats, split='test')

    # Print the per-class results
    results = error_logger.get_errors('test')
    print('\n==================== Test Set Results (epoch %d) ====================' % which_epoch)
    print('Config: %s' % config_path)
    for key, value in results.items():
        print('%-15s: %.4f' % (key, value))
    print('=======================================================================\n')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained segmentation checkpoint on the test split')
    parser.add_argument('-c', '--config', help='training config file used for this checkpoint', required=True)
    parser.add_argument('-e', '--which_epoch', type=int, help='epoch checkpoint to load, e.g. 50', required=True)
    args = parser.parse_args()

    evaluate(args.config, args.which_epoch)
