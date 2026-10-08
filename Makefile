.PHONY: up build clean

up:
	cargo run --bin rusty_rich
	
up-fast:
	cargo run --release --bin rusty_rich

build:
	cargo build --release --bins

clean:
	cargo clean

image:
	docker build -t rusty_rich:latest .

docker-run:
	docker run -p 8080:8080 -it --rm --name rusty_rich_container rusty_rich:latest

docker-build-amd64:
	docker buildx build --platform linux/amd64 -t benmartin94/rusty_rich:latest .
