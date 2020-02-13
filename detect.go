package main

import (
	. "./common"
	"bufio"
	"bytes"
	"flag"
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"image"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
)

// Some constants specific to the pre-trained model at:
// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
// - The model was trained after with images scaled to 224x224 pixels.
// - The colors, represented as R, G, B in 1-byte each were converted to
//   float using (value - Mean)/Scale.
const (
	H, W = 300, 300
)

func main() {
	modelfile := flag.String("model", "", "Path to the trained model")
	labelfile := flag.String("labels", "labels.txt", "Path of a class mapping dict")
	imagefile := flag.String("image", "", "Path of a JPEG-image to extract labels for")
	debugmode := flag.Bool("debug", false, "Enable debug mode")
	minbounds := flag.Float64("min", 0.0, "Minimum confidence to output (WARNING: Will impact ppc)")

	flag.Parse()
	if *modelfile == "" || *imagefile == "" || *labelfile == "" {
		flag.Usage()
		return
	}

	model, err := ioutil.ReadFile(*modelfile)
	if err != nil {
		log.Fatal(err)
	}
	file, err := os.Open(*imagefile)
	if err != nil {
		log.Fatalf("%v", err)
	}

	im, err := jpeg.Decode(file)
	if err != nil {
		log.Fatalf("%s: %v\n", *imagefile, err)
	}

	//
	// all files are open, fire up TF
	//

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// width-number and height-number
	// TODO;; this leaves an offset that is not included
	wn := im.Bounds().Dx() / W
	hn := im.Bounds().Dy() / H

	chips := make([]Chip, wn*hn)
	for i := 0; i < wn*hn; i++ {
		x := i % wn
		y := i / wn
		w := x * W
		h := y * H

		chip := im.(interface {
			SubImage(r image.Rectangle) image.Image
		}).SubImage(image.Rect(w, h, w+W, h+H))

		chips[i] = Chip{x, y, chip}
	}

	if *debugmode {
		writeChips(chips)
	}

	for _, chip := range chips {
		buf := bytes.Buffer{}
		jpeg.Encode(&buf, chip.Im, nil)

		tensor, err := loadImageTensor(buf.Bytes())
		if err != nil {
			log.Fatal(err)
		}
		output, err := session.Run(
			map[tf.Output]*tf.Tensor{
				graph.Operation("image_tensor").Output(0): tensor,
			},
			[]tf.Output{
				graph.Operation("detection_boxes").Output(0),
				graph.Operation("detection_scores").Output(0),
				graph.Operation("detection_classes").Output(0),
				graph.Operation("num_detections").Output(0),
			},
			nil)
		if err != nil {
			log.Fatal(err)
		}

		boxes := output[0].Value().([][][]float32)[0]
		scores := output[1].Value().([][]float32)[0]
		classes := output[2].Value().([][]float32)[0]

		detects := make([]Detect, 1)
		for i, score := range scores {
			class := classes[i]
			detects = append(detects,
				Detect{
					Bounds:     transformBox(chip.X, chip.Y, boxes[i]),
					Class:      CID(class),
					Chip:       &chip,
					Confidence: score,
				})
		}
		printDetections(detects, *labelfile, float32(*minbounds))
	}
}

func transformBox(chipX, chipY int, box []float32) image.Rectangle {
	//     chip pos   ->  world pos
	mx := int(box[1]*W) + (chipX * W)
	Mx := int(box[3]*W) + (chipX * W)
	my := int(box[0]*H) + (chipY * H)
	My := int(box[2]*H) + (chipY * H)

	return image.Rectangle{
		Min: image.Point{X: mx, Y: my,},
		Max: image.Point{X: Mx, Y: My,},
	}
}

func printDetections(detects []Detect, labelsFile string, min float32) {
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}

	labels := make(map[int]string)
	for scanner.Scan() {
		splits := strings.Split(scanner.Text(), ":")
		id, _ := strconv.Atoi(splits[0])
		labels[id] = splits[1]
	}

	for _, d := range detects {
		// squeeze is default; eliminating the 0 entries that inflate ppc
		if d.Confidence > min {
			fmt.Printf("%v %v %v %v %v %v\n", d.Bounds.Min.X, d.Bounds.Min.Y, d.Bounds.Max.X, d.Bounds.Max.Y, d.Class, d.Confidence)
		}
	}
}

func loadImageTensor(im []byte) (*tf.Tensor, error) {
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(im))
	if err != nil {
		return nil, err
	}

	// Construct a graph to normalize the image
	graph, input, output, err := prepareImageTensor()
	if err != nil {
		return nil, err
	}
	// Execute that graph to normalize this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func prepareImageTensor() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	// inception 4D tensor of shape
	// [BatchSize, Height, Width, Colors=3]
	// https://github.com/DIUx-xView/xview2018-baseline/blob/master/inference/det_util.py#L39
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))

	graph, err = s.Finalize()
	return graph, input, output, err
}

func writeChips(chips []Chip) {
	for i, chip := range chips {
		outputFile, _ := os.Create(fmt.Sprintf("/tmp/chip-%v.jpg", i))
		jpeg.Encode(outputFile, chip.Im, &jpeg.Options{Quality: 100})
		outputFile.Close()
	}
}
