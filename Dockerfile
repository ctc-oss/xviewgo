FROM golang:1.12.13-buster as build

ARG PROCESSOR=cpu
ARG TFVERSION=1.15.0
ARG PROTOC=3.11.3

RUN apt update \
 && apt install unzip

WORKDIR /tmp

RUN curl -kL -o protoc.zip \
      https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC}/protoc-${PROTOC}-linux-x86_64.zip

RUN curl -kL -o tf.tar.gz \
      https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${PROCESSOR}-linux-x86_64-${TFVERSION}.tar.gz

RUN unzip protoc.zip \
 && tar xvzf tf.tar.gz \
 && cp -r bin/*     /usr/local/bin \
 && cp -r lib/*     /usr/local/lib \
 && cp -r include/* /usr/local/include \
 && ldconfig

# https://github.com/tensorflow/tensorflow/issues/35133
RUN go get -d github.com/tensorflow/tensorflow/tensorflow/go \
  ; du -sh $GOPATH/src/github.com/tensorflow/tensorflow/tensorflow/go

RUN cd $GOPATH/src/github.com/tensorflow/tensorflow/tensorflow/go \
 && git checkout v${TFVERSION} \
 && go generate github.com/tensorflow/tensorflow/tensorflow/go/op

COPY $PWD /go/src/github.com/jw3/example-tensorflow-golang
WORKDIR   /go/src/github.com/jw3/example-tensorflow-golang

# todo;; real package management
RUN go get "github.com/fogleman/gg" \
 && go get "golang.org/x/image/colornames"

RUN make all \
 && mkdir /tmp/dist \
 && cp dist/* /tmp/dist \
 && cp labels.txt /tmp

#------------------------

FROM centos:7

WORKDIR /opt/workspace

COPY --from=build /tmp/lib/        /usr/local/lib/
COPY --from=build /tmp/dist        /usr/local/bin/
COPY --from=build /tmp/labels.txt  /opt/workspace

ENV LD_LIBRARY_PATH=/usr/local/lib

USER 1001
