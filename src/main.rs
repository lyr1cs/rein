mod config;
mod embed;
mod extract;
mod search;
mod store;
mod types;

fn main() {
    println!("rein v{}", env!("CARGO_PKG_VERSION"));
}
