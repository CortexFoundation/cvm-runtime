rpc-build:
	python -m grpc_tools.protoc \
		-I. --python_out=. --grpc_python_out=. \
		./python/rpc/service.proto
	python ./python/rpc/service.py
rpc-test:
	python ./python/rpc/test_rpc.py
web-build:
	python manage.py runserver 8000
