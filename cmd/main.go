package main

import (
	"log"
	"mlp/internal/app"
)

func main() {
	app := app.NewApp()
	if err := app.Run(); err != nil {
		log.Fatal(err)
	}
}
