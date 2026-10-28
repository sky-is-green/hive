# gVisor for the Hive Docker sandbox

gVisor is a user-space application kernel. `runsc` runs each container through
a separate Sentry process that intercepts syscalls, so the container does not
talk directly to the host kernel. It is an OCI runtime; the Hive image,
workspace mount, `--network none`, labels, and the `/v1/sandbox/exec` path stay
the same. The child does not need a GPU, so gVisor's GPU limitations do not
matter here.

## Install

From the hivebench checkout:

```bash
sudo bash deploy/gvisor/install.sh
```

The script:

1. downloads the `runsc` release and verifies its SHA-512
2. installs `/usr/local/bin/runsc`
3. adds a `runsc` runtime to the rootless daemon config at
   `/srv/hive-ai/home/.config/docker/daemon.json` with
   `runtimeArgs: ["--rootless", "--ignore-cgroups"]` when supported
4. restarts `docker.service` in the `hive-ai` user manager
5. switches the Hive sandbox policy with
   `hive sandbox policy --runtime runsc`
6. runs a real workspace smoke test through `/v1/sandbox/exec`

The daemon config backup is
`/srv/hive-ai/home/.config/docker/daemon.json.pre-gvisor`.

If the smoke test fails, the script sets the runtime policy back to the daemon
default (`runc`). The `runsc` binary and daemon registration remain, so you can
debug and retry. To revert the policy later:

```bash
sudo -u hive-ai env HOME=/srv/hive-ai/home \
  HIVE_HOME=/srv/hive-ai HIVE_STATE_DIR=/srv/hive-ai/state \
  /srv/hive-ai/venv/bin/hive sandbox policy --runtime ""
```

## Why rootless flags

The Hive daemon is rootless Docker under the `hive-ai` user. `runsc` therefore
gets `--rootless`; `--ignore-cgroups` is added when this `runsc` build exposes
it, because the rootless daemon cannot manage the host cgroup hierarchy the
same way a system daemon can. The Docker memory/CPU/pids settings remain in the
policy; the enforcement depends on the rootless cgroup support available on
this host.

## Verify

After the script:

```bash
sudo -u hive-ai env HOME=/srv/hive-ai/home \
  HIVE_HOME=/srv/hive-ai HIVE_STATE_DIR=/srv/hive-ai/state \
  /srv/hive-ai/venv/bin/hive sandbox status
```

The `runtime` line should say `runsc`. Then run the delegate proof; the child
output should still show `/workspace`, `CHILD_OK`, `STATE_ABSENT`,
`SOCKET_ABSENT`, `HOME_ABSENT`, `SIBLING_ABSENT`, `NET_BLOCKED`, and
`CHILD_DONE`.

## Kata Containers

Kata runs a lightweight VM with a guest kernel. The host is KVM-capable
(`kvm_amd` loaded, AMD SVM, `/dev/kvm` present), but the rootless Docker plus
Kata path is not a standard supported deployment: Kata normally expects a
system-level runtime and VM/network management. A system Docker would require
the AI user to access a privileged socket, which conflicts with the
no-docker-group rule. A dedicated VM is the cleaner Kata-class boundary.

`/dev/kvm` is currently mode `0666` root:kvm. gVisor does not need it; if you
pursue Kata or another VM runtime, restrict it to a dedicated group first.

## Boundary

gVisor reduces host-kernel attack surface; it is not a formal proof. It still
shares the host kernel through KVM-free syscall interception and the Sentry
implementation. For a truly adversarial workload, use a dedicated VM or
physical host.
