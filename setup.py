from setuptools import setup

setup(
    name='hoocs',
    version='0.0.1',
    license='GNU LESSER GENERAL PUBLIC LICENSE',
    description='Occlusion-based explainers for higher-order attributions.',
    packages=['hoocs'],  # same as name
    keywords=['XAI', 'PredDiff', 'Shapley values', 'Interactions', 'Model-agnostic attributions'],
    author='Stefan Bluecher',
    author_email='bluecher@tu-berlin.de',
    # external packages as dependencies
    install_requires=['numpy>1.20', 'scipy', 'cv2'],
)
