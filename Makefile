.PHONY: up build clean

up:
	cargo run --bin rusty_rich

build:
	cargo build --release --bins

clean:
	cargo clean
