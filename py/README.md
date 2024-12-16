[[Go back]](../README.md)

## Installation (only one time)
In this `py` directory, type
```
yarn install
```
(will also call yarn setup)

## Setup (each time you go there)
```
yarn setup
```

## Compile (as many time as necessary)
```sh
yarn build
```

## Testing the generated code

Go to the `/cpp-py/bin` folder and type

```sh
python3 test.py
```

## Packaging

### Creating the wheel
```sh
pip3 wheel .
```

### Installing the wheel
```sh
pip3 install THE-GENERATED-WHEEL-NAME.whl --force-reinstall
```

### Testing the installation
If your are still in the `py` folder
```sh
python3 ../bin/test.py
```

### Displaying package info
Informations are gather from the pyproject.toml file.
```sh
pip3 show pyalgo
```

See [this link](https://pybind11.readthedocs.io/en/stable/compiling.html#modules-with-cmake) and [this link](https://scikit-build-core.readthedocs.io/en/latest/) for more informations.
