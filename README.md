go tensorflow
===

example tensorflow object detection inference with golang bindings

targeting the xview 2018 dataset and pretrained inception_v2 models

see the xview 2018 baseline inference scrips for reference that guided this implementation

### Install TensorFlow for Go

Instructions here https://www.tensorflow.org/install/lang_go

... but those are broken:

```shell script
go get github.com/tensorflow/tensorflow/tensorflow/go
```
produces
```text
package github.com/tensorflow/tensorflow/tensorflow/go/genop/internal/proto/github.com/tensorflow/tensorflow/tensorflow/go/core: cannot find package "github.com/tensorflow/tensorflow/tensorflow/go/genop/internal/proto/github.com/tensorflow/tensorflow/tensorflow/go/core" in any of:
        /usr/local/go/src/github.com/tensorflow/tensorflow/tensorflow/go/genop/internal/proto/github.com/tensorflow/tensorflow/tensorflow/go/core (from $GOROOT)
        /go/src/github.com/tensorflow/tensorflow/tensorflow/go/genop/internal/proto/github.com/tensorflow/tensorflow/tensorflow/go/core (from $GOPATH)
```

```shell script
cd $GOPATH/src/github.com/tensorflow/tensorflow/tensorflow/go
git checkout r1.11
git describe --tag
# v1.11.0
cd -
go get github.com/tensorflow/tensorflow/tensorflow/go
go test github.com/tensorflow/tensorflow/tensorflow/go
# ok      github.com/tensorflow/tensorflow/tensorflow/go  (cached)
```

and use the 1.11 C library instead of latest
- https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.11.0.tar.gz
- https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.11.0.tar.gz


### reference
- https://github.com/tensorflow/tensorflow/issues/35133
- https://github.com/tensorflow/tensorflow/issues/34580
- https://stackoverflow.com/a/59453744
- https://hub.packtpub.com/object-detection-go-tensorflow/
- https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/example_inception_inference_test.go
- https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
- https://github.com/DIUx-xView/xview2018-baseline/tree/master/inference


### errors
-`Expects arg[0] to be uint8 but float is provided`
  - https://github.com/tensorflow/models/issues/1741#issuecomment-317501641
