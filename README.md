# IB++ Programming Language

A simple interpreted programming language inspired by BASIC, implemented in Python.

---

## Features

- Variables, arithmetic, and assignment
- If/elif/else, while, and for-like loops
- Functions (methods) with return values
- Lists, strings, and built-in methods
- Input/output and standard library functions
- Error reporting with code context

---

## Getting Started

### 1. Requirements

- Python 3.7 or newer

### 2. Running a Program

```sh
python language.py your_program.pseudo
```

Or, if you make `language.py` executable (see below):

```sh
./language.py your_program.pseudo
```

### 3. Example

Create a file `hello.pseudo`:

```pseudo
output "Hello, IB++!"
```

Run it:

```sh
python language.py hello.pseudo
```

---

## Syntax

See [`syntax.md`](syntax.md) for a full language reference.

---

## Testing

A comprehensive test suite is provided in `tests.pseudo`. Run it with:

```sh
python language.py tests.pseudo
```

---

## Making the Interpreter Executable

On Unix-like systems (Linux, macOS):

1. Ensure the first line of `language.py` is:
    ```python
    #!/usr/bin/env python3
    ```
2. Make it executable:
    ```sh
    chmod +x language.py
    ```
3. Now you can run:
    ```sh
    ./language.py your_program.pseudo
    ```

On Windows, you can always run with:
```sh
python language.py your_program.pseudo
```

---

## Help

Please contact me at: mruiz{at}lincoln{dot}edu{dot}ar
