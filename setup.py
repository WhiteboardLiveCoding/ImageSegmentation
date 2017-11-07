from setuptools import setup

setup(
    name='image_segmentation',
    packages=['image_segmentation'],
    version='0.1',
    description='Segments an image into separate characters',
    author='WhiteboardLiveCoding',
    url='https://github.com/WhiteboardLiveCoding/ImageSegmentation',
    download_url='https://github.com/WhiteboardLiveCoding/ImageSegmentation.git',
    keywords=['ocr', 'segmentation'],
    license='MIT',
    install_requires=[
        'numpy',
        'opencv-python'
    ],
)
