use std::sync::Arc;

use rein::config::ReinConfig;
use rein::ops::OpsRuntime;

#[test]
fn runtime_dry_run_defaults_to_false_and_is_settable() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let mut config = ReinConfig::default();
    config.database.path = tmp.path().join("memories.db").to_string_lossy().into_owned();
    let cfg = Arc::new(config);
    let rt = OpsRuntime::for_cli(cfg);
    assert!(!rt.dry_run(), "default should be false");
    rt.set_dry_run(true);
    assert!(rt.dry_run(), "after set_dry_run(true), should be true");
    rt.set_dry_run(false);
    assert!(!rt.dry_run(), "resettable to false");
}
