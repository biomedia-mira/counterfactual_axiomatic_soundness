from pathlib import Path

import tensorflow as tf
from jax.example_libraries import optimizers

from components.style_gan.style_gan import style_gan
from datasets.celeba_mask_hq import mustache_goatee_scenario
from train import train
from utils import to_numpy_iterator

tf.config.experimental.set_visible_devices([], 'GPU')

if __name__ == '__main__':
    job_dir = Path('/tmp/style_gan_test')
    data_dir = Path('/vol/biomedic/users/mm6818/projects/grand_canyon/data')
    batch_size = 8
    lr = 0.0025
    scenario = mustache_goatee_scenario(data_dir)
    job_name = Path(f'/tmp/style_gan_test')
    scenario_name = 'mustache_goatee_scenario'
    pseudo_oracle_dir = job_dir / scenario_name / 'pseudo_oracles'
    experiment_dir = job_dir / scenario_name / job_name

    train_datasets, test_dataset, parent_dims, is_invertible, marginals, input_shape = scenario

    source_dist = frozenset()
    target_dist = frozenset()
    train_data = to_numpy_iterator(train_datasets[source_dist], batch_size=batch_size)
    test_data = to_numpy_iterator(test_dataset, batch_size=batch_size, drop_remainder=False)

    params = train(model=style_gan(resolution=128),
                   job_dir=job_dir,
                   seed=1,
                   train_data=train_data,
                   test_data=test_data,
                   input_shape=input_shape,
                   optimizer=optimizers.adam(step_size=lr, b1=0.0, b2=.99),
                   num_steps=20000,
                   log_every=100,
                   eval_every=500,
                   save_every=1000,
                   overwrite=True)
