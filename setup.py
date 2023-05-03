from setuptools import setup

setup(
    name='PESTO',
    version='1.0',
    description='Efficient pitch estimation with self-supervised learning',
    author='Alain Riou',
    author_email='alain',
    packages=['PESTO'],
    install_requires=['numpy', 'matplotlib', 'torch', 'torchaudio'],
    classifiers=[  # TODO: no idea what it means, check that
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
