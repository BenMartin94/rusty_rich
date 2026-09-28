use rusty_rich::api;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = api::app();
    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await.expect("failed to bind");
    tracing::info!("rusty_rich API listening on http://{addr}");
    axum::serve(listener, app).await.expect("server error");
}
