from setuptools import setup, find_packages

setup(
    name='pesto',
    version='1.0',
    description='Efficient pitch estimation with self-supervised learning',
    author='Alain Riou',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'pesto': ['weights/*'],  # To include the .pth
    },
    install_requires=[
        'numpy==1.21.5',
        'numpy==1.21.5',
        'torch==2.0.1',
        'torchaudio==2.0.2',
        'nnAudio==0.3.2'
    ],
    classifiers=[
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # If licence is provided must be on the repository
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'pesto=pesto.main:pesto',  # For the command line, executes function pesto() in pesto/main as 'pesto'
        ],
    },
)
