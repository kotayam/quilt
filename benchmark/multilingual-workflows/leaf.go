package main

import (
	"fmt"
	"io"
	"os"
)

const maxPayload = 4096

func main() {
	payload, err := io.ReadAll(io.LimitReader(os.Stdin, maxPayload+1))
	if err != nil || len(payload) > maxPayload {
		os.Exit(1)
	}

	var hash uint32 = 2166136261
	for _, b := range payload {
		if b < ' ' || b > '~' {
			os.Exit(1)
		}
		hash = (hash ^ uint32(b)) * 16777619
	}

	if _, err := fmt.Fprintf(os.Stdout, "%08x\n", hash); err != nil {
		os.Exit(1)
	}
}
