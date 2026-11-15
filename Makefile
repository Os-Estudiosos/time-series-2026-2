setup:
	uv sync
	uv pip install torch --torch-backend=auto

setup-cpu:
	uv sync
	uv pip install torch --torch-backend=cpu