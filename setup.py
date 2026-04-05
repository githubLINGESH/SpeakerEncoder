"""
Setup configuration for Speaker Encoder Pipeline package
"""
from setuptools import setup, find_packages
import os

# Read version from version.txt
version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
if os.path.exists(version_file):
    with open(version_file, 'r') as f:
        version = f.read().strip()
else:
    version = '0.1.0'

# Read long description from README
long_description = ""
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

# Read requirements
requirements = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='speaker-encoder-pipeline',
    version=version,
    description='Multilingual Speaker Encoder with Multi-branch Architecture for Few-shot Voice Synthesis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/Zero_shotVoiceClone',
    license='MIT',
    packages=find_packages(exclude=['tests', 'tests.*', 'wandb', 'models']),
    python_requires='>=3.9',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
    ],
    keywords='speaker-encoder voice-synthesis multilingual few-shot',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/Zero_shotVoiceClone/issues',
        'Documentation': 'https://github.com/yourusername/Zero_shotVoiceClone',
        'Source Code': 'https://github.com/yourusername/Zero_shotVoiceClone',
    },
    entry_points={
        'console_scripts': [
            'speaker-encoder-prepare=prepare_data:main',
            'speaker-encoder-train=train:main',
            'speaker-encoder-validate=validate_speakers:main',
            'speaker-encoder-evaluate=evaluate:main',
        ],
    },
)
