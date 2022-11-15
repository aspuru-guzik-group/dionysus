import tensorflow as tf
from tqdm import tqdm

from . import enums, datasets, models


def get_graphnet_embeddings(work_dir, dataset, task_index=0, verbose=True):
    ''' Train graphnet with final prediction layer for embeddings
    '''
    _, config, _ = datasets.load_dataset(dataset, work_dir)
    config.task = config.tasks[task_index]
    split_type = 'random' if config.task == enums.TaskType.regression else 'stratified'
    smi_split, x, y = datasets.load_task(dataset, enums.FeatureType.graphnet, enums.Models.gnngp, work_dir,
                                         task_index=task_index)
    output_dim = y.values.shape[-1]
    n = y.values.shape[0]

    # get class weights if binary classification task
    if config.task == enums.TaskType.binary:
        y_train, y_val, y_test = y.train, y.val, y.test
        # w_train, w_val, w_test = datasets.get_sample_weights(y_train, y_val, y_test)
        w_train, w_val, w_test = None, None, None  # remove weights
    elif config.task == enums.TaskType.regression:
        y_train, y_val, y_test = y.scaled_train, y.scaled_val, y.scaled_test
        w_train, w_val, w_test = None, None, None
    else:
        raise NotImplementedError()

    # create the model
    hp = models.gnn.default_hp(config.task, output_dim)
    model = models.gnn.GNN.from_hparams(hp)
    optimizer = tf.keras.optimizers.Adam(hp.lr)
    loss_fn = models.training.task_to_loss(hp.task)
    main_metric, metric_fn = models.training.task_to_metric(hp.task)

    # train the model
    early_stop = models.training.EarlyStopping(model, patience=hp.patience)
    stop_metric = f'val_{main_metric}'
    pbar = tqdm(range(hp.epochs), disable=not verbose)
    stats = []

    for _ in pbar:
        models.training.train_step(model, x.train, y_train, optimizer, loss_fn, hp.batch_size, sample_weight=w_train)

        result = {}
        for inputs, target, weight, prefix in [
            (x.train, y_train, w_train, 'train'),
            (x.val, y_val, w_val, 'val'),
            (x.test, y_test, w_test, 'test')
        ]:
            output = model(inputs)
            result[f'{prefix}_{main_metric}'] = metric_fn(target, output, sample_weight=weight)
            result[f'{prefix}_loss'] = loss_fn(target, output, sample_weight=weight).numpy()
        stats.append(result)

        pbar.set_postfix(stats[-1])
        if early_stop.check_criteria(stats[-1][stop_metric]):
            break

    early_stop.restore_best()
    best_step = early_stop.best_step
    print(f'Early stopped at {best_step} with {stop_metric}={stats[best_step][stop_metric]:.3f}')

    embeddings = model.embed(x.values).numpy()
    return embeddings


# DEBUGGING
if __name__ == '__main__':
    pass
