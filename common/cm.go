//
// modified version of https://github.com/sjwhitworth/golearn/blob/master/evaluation/confusion.go
//

package common

import (
	"bytes"
	"fmt"
	"strconv"
	"text/tabwriter"
)

// map truth to predicted and count
type ConfusionMatrix map[CID]map[CID]int

// GetConfusionMatrix builds a ConfusionMatrix from a set of reference (`ref')
// and generate (`gen') Instances.
func GetConfusionMatrix(ref []Truth, gen map[TID]Match, fp map[CID]int) (ConfusionMatrix, error) {
	ret := make(ConfusionMatrix)

	for _, t := range ref {
		if _, ok := ret[t.Class]; !ok {
			ret[t.Class] = make(map[CID]int)
		}
	}

	// FPs; actual is nothing, to predicted c
	ret[0] = make(map[CID]int)
	for c, v := range fp {
		ret[0][c] = v
	}

	// TPs; truth to predicted
	for _, m := range gen {
		ret[m.T.Class][m.D.Class]++
	}

	return ret, nil
}

// GetTruePositives returns the number of times an entry is
// predicted successfully in a given ConfusionMatrix.
func GetTruePositives(class CID, c ConfusionMatrix) float64 {
	return float64(c[class][class])
}

// GetFalsePositives returns the number of times an entry is
// incorrectly predicted as having a given class.
func GetFalsePositives(class CID, c ConfusionMatrix) float64 {
	ret := 0
	for k := range c {
		if k != class {
			ret += c[k][class]
		}
	}
	ret += c[0][class]
	return float64(ret)
}

// GetFalseNegatives returns the number of times an entry is
// incorrectly predicted as something other than the given class.
func GetFalseNegatives(class CID, c ConfusionMatrix) float64 {
	ret := 0
	for k := range c[class] {
		if k != class {
			ret += c[class][k]
		}
	}
	return float64(ret)
}

// GetTrueNegatives returns the number of times an entry is
// correctly predicted as something other than the given class.
func GetTrueNegatives(class CID, c ConfusionMatrix) float64 {
	ret := 0
	for k := range c {
		if k != class {
			for l := range c[k] {
				if l != class {
					ret += c[k][l]
				}
			}
		}
	}
	return float64(ret)
}

// GetPrecision returns the fraction of of the total predictions
// for a given class which were correct.
func GetPrecision(class CID, c ConfusionMatrix) float64 {
	// Fraction of retrieved instances that are relevant
	truePositives := GetTruePositives(class, c)
	falsePositives := GetFalsePositives(class, c)
	return truePositives / (truePositives + falsePositives)
}

// GetRecall returns the fraction of the total occurrences of a
// given class which were predicted.
func GetRecall(class CID, c ConfusionMatrix) float64 {
	// Fraction of relevant instances that are retrieved
	truePositives := GetTruePositives(class, c)
	falseNegatives := GetFalseNegatives(class, c)
	return truePositives / (truePositives + falseNegatives)
}

// GetF1Score computes the harmonic mean of precision and recall
// (equivalently called F-measure)
func GetF1Score(class CID, c ConfusionMatrix) float64 {
	precision := GetPrecision(class, c)
	recall := GetRecall(class, c)
	return 2 * (precision * recall) / (precision + recall)
}

// GetAccuracy computes the overall classification accuracy
// That is (number of correctly classified instances) / total instances
func GetAccuracy(c ConfusionMatrix) float64 {
	correct := 0
	total := 0
	for i := range c {
		for j := range c[i] {
			if i == j {
				correct += c[i][j]
			}
			total += c[i][j]
		}
	}
	return float64(correct) / float64(total)
}

// GetMicroPrecision assesses Classifier performance across
// all classes using the total true positives and false positives.
func GetMicroPrecision(c ConfusionMatrix) float64 {
	truePositives := 0.0
	falsePositives := 0.0
	for k := range c {
		truePositives += GetTruePositives(k, c)
		falsePositives += GetFalsePositives(k, c)
	}
	return truePositives / (truePositives + falsePositives)
}

// GetMacroPrecision assesses Classifier performance across all
// classes by averaging the precision measures achieved for each class.
func GetMacroPrecision(c ConfusionMatrix) float64 {
	precisionVals := 0.0
	for k := range c {
		precisionVals += GetPrecision(k, c)
	}
	return precisionVals / float64(len(c))
}

// GetMicroRecall assesses Classifier performance across all
// classes using the total true positives and false negatives.
func GetMicroRecall(c ConfusionMatrix) float64 {
	truePositives := 0.0
	falseNegatives := 0.0
	for k := range c {
		truePositives += GetTruePositives(k, c)
		falseNegatives += GetFalseNegatives(k, c)
	}
	return truePositives / (truePositives + falseNegatives)
}

// GetMacroRecall assesses Classifier performance across all classes
// by averaging the recall measures achieved for each class
func GetMacroRecall(c ConfusionMatrix) float64 {
	recallVals := 0.0
	for k := range c {
		recallVals += GetRecall(k, c)
	}
	return recallVals / float64(len(c))
}

// GetSummary returns a table of precision, recall, true positive,
// false positive, and true negatives for each class for a given
// ConfusionMatrix
func GetSummary(c ConfusionMatrix, gt map[CID]int) string {
	var buffer bytes.Buffer
	w := new(tabwriter.Writer)
	w.Init(&buffer, 0, 8, 0, '\t', 0)

	fmt.Fprintln(w, "Reference Class\tTruth\tTrue Positives\tFalse Positives\tTrue Negatives\tPrecision\tRecall\tF1 Score")
	fmt.Fprintln(w, "---------------\t-----\t--------------\t---------------\t--------------\t---------\t------\t--------")

	for k := range c {
		t := gt[k]
		tp := GetTruePositives(k, c)
		fp := GetFalsePositives(k, c)
		tn := GetTrueNegatives(k, c)
		prec := GetPrecision(k, c)
		rec := GetRecall(k, c)
		f1 := GetF1Score(k, c)
		fmt.Fprintf(w, "%v\t%v\t%.0f\t%.0f\t%.0f\t%.3f\t%.3f\t%.4f\n", k, t, tp, fp, tn, prec, rec, f1)
	}
	w.Flush()
	buffer.WriteString(fmt.Sprintf("Overall accuracy: %.4f\n", GetAccuracy(c)))

	return buffer.String()
}

// ShowConfusionMatrix return a human-readable version of a given
// ConfusionMatrix.
func ShowConfusionMatrix(c ConfusionMatrix) string {
	var buffer bytes.Buffer
	w := new(tabwriter.Writer)
	w.Init(&buffer, 0, 8, 0, '\t', 0)

	ref := make([]string, 0)
	fmt.Fprintf(w, "Reference Class\t")
	for k := range c {
		fmt.Fprintf(w, "%s\t", k)
		ref = append(ref, string(k))
	}
	fmt.Fprintf(w, "\n")

	fmt.Fprintf(w, "---------------\t")
	for _, v := range ref {
		for t := 0; t < len(v); t++ {
			fmt.Fprintf(w, "-")
		}
		fmt.Fprintf(w, "\t")
	}
	fmt.Fprintf(w, "\n")

	for _, v := range ref {
		fmt.Fprintf(w, "%s\t", v)
		for _, v2 := range ref {
			vv, _ := strconv.Atoi(v)
			vv2, _ := strconv.Atoi(v2)
			fmt.Fprintf(w, "%d\t", c[CID(vv)][CID(vv2)])
		}
		fmt.Fprintf(w, "\n")
	}
	w.Flush()

	return buffer.String()
}
