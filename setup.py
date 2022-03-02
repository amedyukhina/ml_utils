from setuptools import setup

setup(
    name='ml_utils',
    version='0.1.0',
    url="https://https://github.com/amedyukhina/bbox_detection",
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['ml_utils',
              'ml_utils.dataset',
              'ml_utils.model',
              'ml_utils.train',
              'ml_utils.utils',
              'ml_utils.transforms',
              'ml_utils.predict'],
    license='Apache License Version 2.0',
    include_package_data=True,

    test_suite='tests',

    install_requires=[
        'ipykernel',
        'wandb',
        'torch>=1.10',
        'torchvision >=0.11',
        'albumentations',
        'matplotlib >=3.0.3',
        'scipy',
        'numpy',
        'ddt',
        'pytest',
        'tqdm',
        'scikit-image',
        'pandas',
        'jupyter',
        'am_utils @ git+https://github.com/amedyukhina/am_utils.git',
    ],
)
