package common

import "image"

type TID int
type DID int
type CID int

type Chip struct {
	X  int
	Y  int
	Im image.Image
}

type Truth struct {
	Id     TID
	Bounds image.Rectangle
	Class  CID
}

type Detect struct {
	Id         DID
	Bounds     image.Rectangle
	Class      CID
	Chip       *Chip
	Confidence float32
}

type Match struct {
	T   Truth
	D   Detect
	IoU float32
}
