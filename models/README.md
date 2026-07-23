`models` contains the [.h5](https://www.tensorflow.org/guide) and [.nnet](https://github.com/sisl/NNet) files used to represent DNNs.

.h5 files are used to verify for Fairify, and .nnet files are used to verify for FairQuant.
All the networks are trained in Tensorflow as .h5 files and were subsequently converted into .nnet files.

- `adult`, `bank`, and `german` contain the TensorFlow .h5 networks provided by Fairify. 
- `compas` contain our newly trained TensorFlow .h5 networks.
- `example.nnet` is the toy network presented in the paper.


`compas-7` files are compressed as .tar.gz files, so please extract them by running the following command:
```shell
cd compas
tar -xzf compas-7.h5.tar.gz 
tar -xzf compas-7.nnet.tar.gz
```
