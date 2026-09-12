#!/usr/bin/env python
"""Local CA + server certificate generation for the phone-control HTTPS server.

Why HTTPS at all: the phone page reads its orientation through Android WebXR
`inline` sessions, and WebXR is only exposed in secure contexts (https:// or
localhost). TLS therefore stays, but both earlier pain points go away:

* The server auto-generates these files when they are missing (see server.py),
  so a missing key.pem/cert.pem never blocks startup.
* Instead of one untrusted self-signed certificate (which phones flag as a
  high-risk site and refuse to trust), this creates a project-local CA and
  signs a server leaf that carries an IP SAN for the machine's current LAN
  address. Install the CA root on the phone once, and https://<lan-ip>:4445
  validates with no warnings at all.

Everything is produced by the system openssl; no network access is needed.
The leaf is limited to 398 days, the longest validity modern browsers accept.
When the machine's LAN address changes, rerun this script (or simply start
the server) — the leaf is refreshed with the new SAN while the phone keeps
trusting the unchanged CA root.

Usage:
    python experiments/phone_orientation_control/tls_certs.py [--force]
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CERT_DIR = THIS_DIR.parent / "phone_pose_viz"
CA_CN = "SO101 phone control local CA"
LEAF_DAYS = 398  # browser-accepted maximum for server certificates
CA_DAYS = 3650
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$")


_SKIPPED_IFACES = ("lo", "docker", "veth", "br-", "virbr", "cni", "flannel", "ppp", "wg")


def detect_ipv4_addresses() -> list[str]:
    """All non-loopback interface IPv4 addresses, best-effort.

    Collecting every address (LAN plus Tailscale's tailscale0, ...) means one
    certificate serves whichever network path the phone actually uses to reach
    the server, e.g. http can reach it on the campus net but not over
    Tailscale, or vice versa. Falls back to the default-route address.
    """
    addresses: list[str] = []
    try:
        proc = subprocess.run(
            ["ip", "-4", "-o", "addr", "show"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = line.split()
                if len(parts) < 4:
                    continue
                iface = parts[1]
                if iface == "lo" or iface.startswith(_SKIPPED_IFACES):
                    continue
                address = parts[3].split("/")[0]
                if address not in addresses:
                    addresses.append(address)
    except (OSError, subprocess.TimeoutExpired):
        pass
    if not addresses:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            addresses.append(str(sock.getsockname()[0]))
        except OSError:
            addresses.append("127.0.0.1")
        finally:
            sock.close()
    return addresses


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()}")


_SAN_RE = re.compile(r"(?:DNS|IP Address):[A-Za-z0-9.:-]+")


def _leaf_sans(cert_path: Path) -> set[str]:
    """Return the subjectAltName entries of an existing leaf, e.g. {"IP Address:10.0.0.5"}."""
    proc = subprocess.run(
        ["openssl", "x509", "-in", str(cert_path), "-noout", "-ext", "subjectAltName"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return set()
    # "IP Address:1.2.3.4" contains a space, so plain split() would mangle it.
    return {match.rstrip(",") for match in _SAN_RE.findall(proc.stdout)}


def _write_config(cert_dir: Path, hosts: list[str], ips: list[str]) -> Path:
    """Write an openssl config carrying the SAN list plus CA/server extension blocks."""
    lines = [
        "[req]",
        "distinguished_name = dn",
        "prompt = no",
        "[dn]",
        "[v3_ca]",
        "basicConstraints = critical,CA:TRUE",
        "keyUsage = critical,keyCertSign,cRLSign",
        "subjectKeyIdentifier = hash",
        "[v3_server]",
        "basicConstraints = critical,CA:FALSE",
        "keyUsage = critical,digitalSignature,keyEncipherment",
        "extendedKeyUsage = serverAuth",
        "subjectAltName = @alt_names",
        "[alt_names]",
    ]
    for index, host in enumerate(hosts, start=1):
        lines.append(f"DNS.{index} = {host}")
    for index, ip in enumerate(ips, start=1):
        lines.append(f"IP.{index} = {ip}")
    path = cert_dir / "openssl_san.cnf"
    path.write_text("\n".join(lines) + "\n")
    return path


def _generate_ca(cert_dir: Path, cfg: Path) -> None:
    _run(
        [
            "openssl", "req", "-x509", "-new", "-newkey", "rsa:2048", "-sha256", "-nodes",
            "-keyout", "rootCA.key", "-out", "rootCA.crt",
            "-days", str(CA_DAYS), "-subj", f"/CN={CA_CN}",
            "-config", str(cfg), "-extensions", "v3_ca",
        ],
        cwd=cert_dir,
    )


def _generate_leaf(cert_dir: Path, cfg: Path, hosts: list[str], ips: list[str]) -> None:
    common_name = hosts[0] if hosts else ips[0]
    _run(
        [
            "openssl", "req", "-new", "-newkey", "rsa:2048", "-sha256", "-nodes",
            "-keyout", "key.pem", "-out", "server.csr",
            "-subj", f"/CN={common_name}",
        ],
        cwd=cert_dir,
    )
    try:
        _run(
            [
                "openssl", "x509", "-req", "-sha256", "-days", str(LEAF_DAYS),
                "-in", "server.csr",
                "-CA", "rootCA.crt", "-CAkey", "rootCA.key", "-CAcreateserial",
                "-out", "cert.pem",
                "-extfile", str(cfg), "-extensions", "v3_server",
            ],
            cwd=cert_dir,
        )
    finally:
        (cert_dir / "server.csr").unlink(missing_ok=True)


def ensure_certificates(
    cert_dir: Path,
    ips: tuple[str, ...] = (),
    hostnames: tuple[str, ...] = ("localhost",),
    force: bool = False,
    log=None,
) -> tuple[Path, Path]:
    """Make sure the CA root and a server leaf with SANs for every IP exist.

    Auto-detects the current LAN IP when none is given. The leaf is renewed
    when it is missing or does not already cover one of the requested IPs
    (LAN addresses change; the CA root stays valid on the phone). ``log``
    accepts a logging.Logger; stdout is used when it is None.
    """
    note = log.info if log is not None else print
    cert_dir.mkdir(parents=True, exist_ok=True)
    hosts = list(hostnames)
    ips = list(ips) or detect_ipv4_addresses()
    ips = list(dict.fromkeys(ips))  # de-duplicate, keep order
    for ip in ips:
        ipaddress.ip_address(ip)  # raises ValueError on garbage
    hosts = [host for host in hosts if host and _HOSTNAME_RE.match(host)]
    if not hosts:
        hosts = ["localhost"]

    ca_key, ca_crt, key, cert = (
        cert_dir / "rootCA.key", cert_dir / "rootCA.crt",
        cert_dir / "key.pem", cert_dir / "cert.pem",
    )
    cfg = _write_config(cert_dir, hosts, ips)
    try:
        if force or not (ca_key.is_file() and ca_crt.is_file()):
            _generate_ca(cert_dir, cfg)
            note(f"created local CA: {ca_crt}")
        existing = _leaf_sans(cert) if cert.is_file() else set()
        wanted = {f"IP Address:{ip}" for ip in ips}
        if force or not (key.is_file() and cert.is_file()) or not wanted <= existing:
            _generate_leaf(cert_dir, cfg, hosts, ips)
            note(f"created server certificate (SAN: {', '.join([*hosts, *ips])})")
    finally:
        cfg.unlink(missing_ok=True)
    for private in (ca_key, key):
        private.chmod(0o600)
    return ca_crt, cert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cert-dir", type=Path, default=DEFAULT_CERT_DIR,
        help="where to keep rootCA.* and the server key.pem/cert.pem (default: %(default)s)",
    )
    parser.add_argument(
        "--ip", action="append", default=[],
        help="LAN IP to add as a certificate SAN; repeatable (default: auto-detect)",
    )
    parser.add_argument("--force", action="store_true", help="regenerate everything")
    args = parser.parse_args()

    ips = tuple(args.ip) or tuple(detect_ipv4_addresses())
    ca_crt, _ = ensure_certificates(args.cert_dir, ips=ips, force=args.force)
    addresses = "\n".join(f"  phone:    https://{ip}:4445" for ip in ips)
    print(
        f"\nCertificates ready in {args.cert_dir}"
        f"\n  CA root:     {ca_crt}"
        f"\n  server leaf: {args.cert_dir / 'cert.pem'} (SAN covers {', '.join(ips)})"
        f"\n\nOnce the server runs, the phone can reach it at:"
        f"\n{addresses}"
        f"\nOn networks that isolate devices (e.g. campus Wi-Fi), open the"
        f"\nTailscale 100.x address instead and use the Tailscale app on the phone."
        f"\n\nOne-time phone trust (Android, same CA for every address):"
        f"\n  1. Open https://<any address above>:4445/ca.crt in the phone browser and"
        f"\n     download rootCA.crt."
        f"\n  2. Android: Settings > Security > More security & privacy > Encryption &"
        f"\n     credentials > Install a certificate > CA certificate, then pick rootCA.crt"
        f"\n     from Downloads (may require the screen lock PIN)."
        f"\n  3. Afterwards every https://<address>:4445 above opens without warnings."
        f"\n\nIf the machine's addresses change later, rerun this script (or start the"
        f"\nserver): the leaf is renewed for the current address set and the phone keeps"
        f"\ntrusting the same CA.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
