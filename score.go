package main

import (
	. "./common"
	"encoding/csv"
	"encoding/json"
	"flag"
	"image"
	"io/ioutil"
	"log"
	"os"
	"strings"
)

type FeatureCollection struct {
	Features []Feature
}

type Feature struct {
	Properties struct {
		Id     int    `json:"feature_id"`
		Bounds string `json:"bounds_imcoords"`
		Class  int    `json:"type_id"`
	}
}

type Stats struct {
	GroundTruthClasses map[CID]int
	AveragePrecision   map[CID]float32
}

func main() {
	pFile := flag.String("predictions", "", "Path to predictions csv")
	tFile := flag.String("groundtruth", "", "Path to ground-truth geojson")
	minIou := flag.Float64("iou", .5, "IOU threshold")
	minConf := flag.Float64("confidence", .5, "Confidence threshold")

	flag.Parse()
	if *pFile == "" || *tFile == "" {
		flag.Usage()
		return
	}

	_, err := ioutil.ReadFile(*pFile)
	if err != nil {
		log.Fatal(err)
	}

	f, _ := os.Open(*pFile)
	csvr := csv.NewReader(f)
	csvr.Comma = ' '
	predictions, _ := csvr.ReadAll()

	tbytes, err := ioutil.ReadFile(*tFile)
	if err != nil {
		log.Fatal(err)
	}

	var ref FeatureCollection
	json.Unmarshal(tbytes, &ref)

	gtc := make(map[CID]int)
	detects := ReadDetects(predictions)

	truth := make([]Truth, len(ref.Features))
	for i, rf := range ref.Features {
		splits := strings.Split(rf.Properties.Bounds, ",")
		truth[i] = Truth{
			Id: TID(rf.Properties.Id),
			Bounds: SplitToRect(splits),
			Class: CID(rf.Properties.Class),
		}

		gtc[CID(rf.Properties.Class)]++
	}

	matched := make(map[TID]Match, len(truth))
	unmatched := make(map[CID]int)
	fds := make([]Detect, 0)

	for _, d := range detects {
		if d.Confidence >= float32(*minConf) {
			found := false
			for _, t := range truth {
				if _, here := matched[t.Id]; !here {
					i := t.Bounds.Intersect(d.Bounds)

					// calculate IOU if overlapping
					if !i.Empty() {
						ia := area(i)
						id := area(d.Bounds)
						it := area(t.Bounds)
						iou := float32(ia) / float32(id+it-ia)

						if iou >= float32(*minIou) {
							matched[t.Id] = Match{T: t, D: d, IoU: iou}
							found = true
							break
						}
					}
				}
			}

			if !found {
				// false-positive due to non intersecting box
				// todo;; could also be a duplicate
				fds = append(fds, d)
				unmatched[d.Class]++
			}
		}
	}

	cm, _ := GetConfusionMatrix(gtc, matched, unmatched)

	println(len(ref.Features))
	println(len(predictions))
	println(GetSummary(cm))
}

func area(r image.Rectangle) int {
	z := r.Size()
	return z.X * z.Y
}
