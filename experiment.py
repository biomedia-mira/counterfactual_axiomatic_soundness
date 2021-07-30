import shutil
from pathlib import Path

from datasets.confounded_mnist import create_confounded_mnist_dataset
from model.model import build_model
from model.train import train

if __name__ == '__main__':

    overwrite = True
    job_dir = Path('/tmp/test_run_5')
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(exist_ok=True, parents=True)
    if job_dir.exists() and not overwrite:
        exit(1)

    mode = 'rgb'
    noise_dim = 10
    train_data, parent_dims, marginals, img_decode_fn, input_shape = \
        create_confounded_mnist_dataset(batch_size=1024, debug=True, jpeg_encode=True if mode == 'jpeg' else False)
    model = build_model(parent_dims, marginals, input_shape, noise_dim, mode, img_decode_fn, cycle=False)
    test_data = None
    train(model=model,
          input_shape=input_shape,
          job_dir=job_dir,
          num_epochs=1000,
          train_data=train_data,
          test_data=None,
          eval_every=10,
          save_every=100)
