from pathlib import Path
from setuptools import setup, find_packages

def get_readme_text():
    root_dir = Path(__file__).parent
    readme_path = root_dir / "README.md"
    return readme_path.read_text()


setup(
    name='pesto-pitch',
    version='0.1.0',
    description='Efficient pitch estimation with self-supervised learning',
    long_description=get_readme_text(),
    long_description_content_type='text/markdown',
    author='Alain Riou',
    url='https://github.com/SonyCSLParis/pesto',
    license='LGPL-3.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'pesto': ['weights/*'],  # To include the .pth
    },
    install_requires=[
        'numpy>=1.21.5',
        'scipy>=1.8.1',
        'tqdm>=4.66.1',
        'torch>=2.0.1',
        'torchaudio>=2.0.2'
    ],
    classifiers=[
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',  # If licence is provided must be on the repository
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={
        'console_scripts': [
            'pesto=pesto.main:pesto',  # For the command line, executes function pesto() in pesto/main as 'pesto'
        ],
    },
    python_requires='>=3.8',
)
