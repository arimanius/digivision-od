# od

## build proto

```shell
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. od/api/v1/*.proto
```
