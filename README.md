OCR Training
=====

This repository holds the script to train the neural network used in
[WhiteboadLiveCoding](https://github.com/WhiteboardLiveCoding/WhiteboardLiveCoding).
It also houses the script to add images to the dataset.

## Usage

#### [training.py](training.py)

    usage: training.py --emnist <path-to-emnist.mat> --wlc <path-to-wlc.mat>

##### Required Arguments:

    --emnist EMNIST       path for the EMNIST dataset
    --wlc WLC             path for the WLC dataset

##### Optional Arguments

    -h, --help            show the help message and exit
    -o OUTPUT, --output OUTPUT
                          output directory for the model(without /)
                          default: bin
    --height HEIGHT       height of the input image
                          default: 28
    --width WIDTH         width of the input image
                          default: 28
    -e EPOCHS, --epochs EPOCHS
                          number of epochs to train the model on
                          default: 10
    -g GPUS, --gpus GPUS  number of gpus to be used
                          default: 1
    -b BATCH, --batch BATCH
                          batch size for training
                          default: 64
    -d DEVICE, --device DEVICE
                        device to be used for training
                        default: /cpu:0
    -m MODEL, --model MODEL
                          keras model to be trained
                          default: convolutional
    -p , --parallel       use multi gpu model
    -f, --fix-emnist      fix the images from emnist


#### [dataset.py](dataset.py)

A Dataset class has been defined to load the `wlc-byclass` dataset
and add more images to it.

```python
    # Reference Code
    from OCRTraining.dataset import Dataset

    dataset = Dataset(batch_size=32) # The new images are cached and put in the dataset in a batch

    # This adds an image to the training data with label '{'
    dataset.add_image([for i in range(784)], '{')

    # This adds an image to the testing data with label '{'
    dataset.add_image([for i in range(784)], '{', test_data=True)

    # This saves the dataset to a .mat file with the timestamp as a part of the filename
    dataset.save()

    # A do_compression=False argument can be supplied to save an uncompressed dataset
    dataset.save(do_compression=False)
```

Based on a project by @coopss
