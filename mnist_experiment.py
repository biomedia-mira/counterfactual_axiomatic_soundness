import shutil
from pathlib import Path

from datasets.mnist_mechanisms import create_confounded_mnist_dataset

if __name__ == '__main__':

    overwrite = False
    job_dir = Path('/tmp/test_3')
    if job_dir.exists() and overwrite:
        shutil.rmtree(job_dir)

    train_datasets, test_dataset, parent_dims, marginals, input_shape = create_confounded_mnist_dataset()

    #
    # classifiers, divergences, mechanisms = build_functions(params, *functions)
    # repeat_test = {p_name + '_repeat': repeat_transform_test(mechanism, p_name, noise_dim, n_repeats=10)
    #                for p_name, mechanism in mechanisms.items()}
    # cycle_test = {p_name + '_cycle': cycle_transform_test(mechanism, p_name, noise_dim, parent_dims[p_name])
    #               for p_name, mechanism in mechanisms.items()}
    # permute_test = {'permute': permute_transform_test({p_name: mechanisms[p_name]
    #                                                    for p_name in ['color', 'thickness']}, parent_dims, noise_dim)}
    # tests = {**repeat_test, **cycle_test, **permute_test}
    # res = perform_tests(test_data, tests)
