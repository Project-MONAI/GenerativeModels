# UNIT TESTS

## Set environment
To use the tests already available at MONAI core, first we clone it:
```shell
git clone https://github.com/Project-MONAI/MONAI --branch main
```

Then we add it to PYTHONPATH
```shell
export PYTHONPATH="${PYTHONPATH}:./MONAI/"
```

## Executing tests
To run tests, use the following command:

```shell script
 python -m unittest discover tests
```
