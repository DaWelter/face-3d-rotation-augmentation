from setuptools import setup, find_packages

# find packages
packages = find_packages(exclude=("test", "test.*"))

# setup
setup(
    name='face-3d-rotation-augmentation',
    description='Generates large pose faces',
    author='Michael Welter',
    license='MIT Licence',
    packages=packages,
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'pyrender'
    ],
    python_requires=">=3.9",
)