import shutil
from pathlib import Path

from datasets.confounded_mnist import create_confounded_mnist_dataset
from model.model import build_model
from model.evaluation import get_evaluation_update_and_log_fns
from trainer.training import train

if __name__ == '__main__':
    overwrite = True
    num_epochs = 100
    job_dir = Path('/tmp/test_run_4')
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(exist_ok=True, parents=True)

    dataset, parent_dims, marginals, decode_fn, seq_shape = create_confounded_mnist_dataset(128, debug=True)
    init_fun, apply_fun, init_optimizer_fun = build_model(parent_dims=parent_dims,
                                                          marginals=marginals,
                                                          seq_shape=seq_shape)
    update_eval, log_eval = get_evaluation_update_and_log_fns(decode_fn)

    if not job_dir.exists() or overwrite:
        train(init_fun=init_fun,
              apply_fun=apply_fun,
              init_optimizer_fun=init_optimizer_fun,
              update_eval=update_eval,
              log_eval=log_eval,
              input_shape=(-1, *seq_shape),
              job_dir=job_dir,
              num_epochs=num_epochs,
              train_data=dataset,
              test_data=None,
              eval_every=10,
              save_every=10)
