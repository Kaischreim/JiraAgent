package main

import "fmt"

func main() {
    fmt.Println("Hello from DependAI!")
}

func Add(a, b int) int {
    return a + b
}

func Greet(name string) string {
    return fmt.Sprintf("Hello, %s!", name)
}
