# encoding: utf-8
# edit : 
# - author : wblee
# - date : 2025-08-28

from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name='pickt',
        version='1.0.0',
        description='PICKT: Practical Interlinked Concept Knowledge Tracing for Personalized Learning using Knowledge Map Concept Relations',
        url='https://github.com/wonbeeny/PICKT.git',
        install_requires=[
            "fastexcel==0.14.0",
            "omegaconf==2.3.0",
            "openpyxl==3.1.5",
            "overrides==7.7.0",
            "pandas==2.2.3",
            "polars==1.29.0",
            "lightning==2.5.1",
            "scikit-learn==1.6.1",
            "tabulate==0.9.0",
            "torch==2.5.0",
            "torch_geometric==2.6.1",
            "transformers==4.46.1",
            "tensorboard==2.19.0",
            "umap-learn==0.5.7"
        ],
        author='WonbeenLee',
        author_email='wonbeeny@gmail.com',
        license='CC BY-NC 4.0',
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: Free for non-commercial use',
            'Operating System :: OS Independent',
        ],
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        zip_safe=False,
        include_package_data=True,
        package_data={'': ['*.pickle']}
    )