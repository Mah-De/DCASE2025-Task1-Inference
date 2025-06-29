from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='Ramezanee-SUT-Task1',
    version='0.1.0',
    description='SUT inference code',
    author='Mohamad Mahdee Ramezanee',
    author_email="mahdramezanee@gmail.com",
    packages=find_packages(),  # This auto-discovers the inner folder
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'Ramezanee-SUT-Task1': ["resources/*.wav", 'ckpts/*.ckpt', 'ckpts/*.*'],
    },
    python_requires='>=3.13',
)
