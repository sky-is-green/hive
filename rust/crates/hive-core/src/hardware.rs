//! Lightweight Linux host probes used by `hive doctor`.
//!
//! The CLI only needs to answer "is this box ready?"; it does not need a full
//! hardware profile. AMD cards come from sysfs in PCI-BDF order, which is the
//! order ROCm/HIP uses for device indices.

use std::collections::BTreeMap;
use std::fs;
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
use std::path::Path;

use crate::python::which;

#[derive(Debug, Clone, Default)]
pub struct Device {
    pub card: String,
    pub bdf: Option<String>,
    pub driver: Option<String>,
    pub vendor: Option<String>,
    pub device: Option<String>,
    pub memory_gb: Option<f64>,
    pub used_gb: Option<f64>,
    pub free_gb: Option<f64>,
    pub gtt_total_gb: Option<f64>,
    pub gtt_used_gb: Option<f64>,
    pub util_pct: Option<i64>,
    pub display: bool,
    pub index: usize,
}

impl Device {
    /// `card 0: 3.83 used / 16.15 free GiB (display)`.
    pub fn detail(&self) -> String {
        let state = match (self.used_gb, self.free_gb) {
            (Some(used), Some(free)) => format!("{used} used / {free} free GiB"),
            _ => format!(
                "{} GiB",
                self.memory_gb.map(|v| v.to_string()).unwrap_or_default()
            ),
        };
        let display = if self.display { " (display)" } else { "" };
        format!("card {}: {state}{display}", self.index)
    }
}

/// Enumerate amdgpu cards from sysfs, sorted by PCI BDF.
pub fn amd_devices() -> std::io::Result<Vec<Device>> {
    let drm = Path::new("/sys/class/drm");
    let mut devices = Vec::new();
    for entry in fs::read_dir(drm)? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(suffix) = name.strip_prefix("card") else {
            continue;
        };
        if suffix.is_empty() || !suffix.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let device_dir = entry.path().join("device");
        if !device_dir.is_dir() {
            continue;
        }
        let read = |name: &str| -> Option<String> {
            fs::read_to_string(device_dir.join(name))
                .ok()
                .map(|text| text.trim().to_owned())
                .filter(|text| !text.is_empty())
        };
        let Some(total) = read("mem_info_vram_total") else {
            continue;
        };
        let mut bdf = None;
        let mut driver = None;
        if let Ok(uevent) = fs::read_to_string(device_dir.join("uevent")) {
            for line in uevent.lines() {
                if let Some(value) = line.strip_prefix("PCI_SLOT_NAME=") {
                    bdf = Some(value.trim().to_owned());
                } else if let Some(value) = line.strip_prefix("DRIVER=") {
                    driver = Some(value.trim().to_owned());
                }
            }
        }
        let display =
            display_connected(&name).unwrap_or_else(|| read("boot_vga").as_deref() == Some("1"));
        let busy = read("gpu_busy_percent");
        let vendor = read("vendor");
        let device = read("device");
        let memory_gb = gb(&total);
        let used_gb = read("mem_info_vram_used").as_deref().and_then(gb);
        devices.push(Device {
            card: name,
            bdf,
            driver,
            vendor,
            device,
            memory_gb,
            used_gb,
            free_gb: match (memory_gb, used_gb) {
                (Some(total), Some(used)) => Some(round2((total - used).max(0.0))),
                _ => None,
            },
            gtt_total_gb: read("mem_info_gtt_total").as_deref().and_then(gb),
            gtt_used_gb: read("mem_info_gtt_used").as_deref().and_then(gb),
            util_pct: busy.as_deref().and_then(|text| text.parse::<i64>().ok()),
            display,
            index: 0,
        });
    }
    devices.sort_by(|a, b| {
        a.bdf
            .as_deref()
            .unwrap_or("")
            .cmp(b.bdf.as_deref().unwrap_or(""))
    });
    for (index, device) in devices.iter_mut().enumerate() {
        device.index = index;
    }
    Ok(devices)
}

fn display_connected(card: &str) -> Option<bool> {
    let drm = Path::new("/sys/class/drm");
    let prefix = format!("{card}-");
    for entry in fs::read_dir(drm).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with(&prefix) {
            continue;
        }
        if let Ok(status) = fs::read_to_string(entry.path().join("status"))
            && status.trim() == "connected"
        {
            return Some(true);
        }
    }
    Some(false)
}

fn gb(raw: &str) -> Option<f64> {
    raw.trim()
        .parse::<u64>()
        .ok()
        .map(|value| round2(value as f64 / (1024.0 * 1024.0 * 1024.0)))
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Host memory `(available, total)` in GiB, as psutil reports it.
pub fn memory_gib() -> Option<(f64, f64)> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    let mut values: BTreeMap<&str, u64> = BTreeMap::new();
    for line in text.lines() {
        let (key, rest) = line.split_once(':')?;
        let value = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        values.insert(key, value);
    }
    let total = *values.get("MemTotal")? as f64 / 1024.0 / 1024.0;
    let available = *values.get("MemAvailable")? as f64 / 1024.0 / 1024.0;
    Some((available, total))
}

/// Free space in GiB for the filesystem holding `path`.
pub fn disk_free_gib(path: &Path) -> std::io::Result<f64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has NUL"))?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let available = stat.f_bavail as f64 * stat.f_frsize as f64;
        Ok(available / (1024.0 * 1024.0 * 1024.0))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(0.0)
    }
}

/// Whether the loopback port can be bound right now.
pub fn port_free(port: u16) -> bool {
    TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).is_ok()
}

/// `rocminfo` or `rocm-smi`, when present.
pub fn rocm_tool() -> Option<std::path::PathBuf> {
    which("rocminfo").or_else(|| which("rocm-smi"))
}

/// `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or a cached token file.
pub fn has_hf_token() -> bool {
    if std::env::var_os("HF_TOKEN").is_some()
        || std::env::var_os("HUGGING_FACE_HUB_TOKEN").is_some()
    {
        return true;
    }
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    home.map(|home| {
        home.join(".cache/huggingface/token").is_file() || home.join(".huggingface/token").is_file()
    })
    .unwrap_or(false)
}
