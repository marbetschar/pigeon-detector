# pigeon-detector
Detecting Pigeons in Images using Machine Learning

## Run detection on single image

```shell
./app.sh --config config-prod.yaml --image <path>
```

## Run continous detection in interactive mode

```shell
./app.sh --config config-prod.yaml
```

## Run continous detection in background

```shell
nohup ./app.sh --config config-prod.yaml --daemon &
```