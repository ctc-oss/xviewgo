go tensorflow
===

example tensorflow object detection inference with golang bindings

targeting the xview 2018 dataset and pretrained inception_v2 models

see the xview 2018 baseline inference scrips for reference that guided this implementation

### run

```shell script
make
detect -model xview-models/multires.pb -image xview/2122.jpg > predictions.txt
score -predictions predictions.txt -groundtruth xview/labels/2122.geojson
```


### Install TensorFlow for Go
- install recent protoc, eg. v3.11.3
- download and install a 1.15.0 lib, one of
  - https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz
  - https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
- ldconfig
- go get -d github.com/tensorflow/tensorflow/tensorflow/go
- cd $GOPATH/src/github.com/tensorflow/tensorflow/tensorflow/go
- git checkout v1.15.0
- go generate github.com/tensorflow/tensorflow/tensorflow/go/op


### reference
- https://www.tensorflow.org/install/lang_go
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README.md
- https://github.com/tensorflow/tensorflow/issues/35133
- https://github.com/tensorflow/tensorflow/issues/34580
- https://stackoverflow.com/a/59453744
- https://hub.packtpub.com/object-detection-go-tensorflow/
- https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/example_inception_inference_test.go
- https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
- https://github.com/DIUx-xView/xview2018-baseline/tree/master/inference
- https://pgaleone.eu/tensorflow/go/2017/05/29/understanding-tensorflow-using-go/


### errors
-`Expects arg[0] to be uint8 but float is provided`
  - https://github.com/tensorflow/models/issues/1741#issuecomment-317501641
  