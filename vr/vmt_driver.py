"""
vr/vmt_driver.py - SteamVR Virtual Motion Tracker (VMT) Driver Setup & Registration.
Discovers SteamVR installations, registers/unregisters the VMT OpenVR driver using vrpathreg
or direct openvrpaths.vrpath configuration, and checks SteamVR driver readiness.
"""

import os
import sys
import json
import subprocess
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def get_openvrpaths_path() -> Path:
    """Returns the path to openvrpaths.vrpath on the current system."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            return Path(local_app_data) / "openvr" / "openvrpaths.vrpath"
        return Path.home() / "AppData" / "Local" / "openvr" / "openvrpaths.vrpath"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "OpenVR" / "openvrpaths.vrpath"
    else:
        return Path.home() / ".config" / "openvr" / "openvrpaths.vrpath"


def read_openvrpaths() -> Optional[Dict[str, Any]]:
    """Reads and parses openvrpaths.vrpath."""
    vrpath = get_openvrpaths_path()
    if vrpath.exists():
        try:
            with open(vrpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"[VMT Setup] Error reading {vrpath}: {exc}")
    return None


def write_openvrpaths(data: Dict[str, Any]) -> bool:
    """Writes openvrpaths.vrpath safely."""
    vrpath = get_openvrpaths_path()
    try:
        vrpath.parent.mkdir(parents=True, exist_ok=True)
        with open(vrpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as exc:
        print(f"[VMT Setup] Error writing {vrpath}: {exc}")
        return False


def find_vrpathreg_binary() -> Optional[Path]:
    """Locates vrpathreg.exe across common Steam and SteamVR directories."""
    paths_to_check = [
        r"C:\Program Files (x86)\Steam\steamapps\common\SteamVR\bin\win64\vrpathreg.exe",
        r"C:\Program Files\Steam\steamapps\common\SteamVR\bin\win64\vrpathreg.exe",
        r"D:\Steam\steamapps\common\SteamVR\bin\win64\vrpathreg.exe",
        r"E:\Steam\steamapps\common\SteamVR\bin\win64\vrpathreg.exe",
    ]

    # Check via openvrpaths runtime directory if available
    vrpaths = read_openvrpaths()
    if vrpaths and "runtime" in vrpaths:
        for r in vrpaths["runtime"]:
            cand = Path(r) / "bin" / "win64" / "vrpathreg.exe"
            if cand.exists():
                return cand

    # Check common fallback locations
    for p in paths_to_check:
        cand = Path(p)
        if cand.exists():
            return cand

    return None


def get_registered_drivers() -> List[str]:
    """Returns a list of external driver directories currently registered in SteamVR."""
    vrpaths = read_openvrpaths()
    if vrpaths and "external_drivers" in vrpaths:
        return vrpaths["external_drivers"]
    return []


def is_vmt_registered(vmt_dir: Optional[str] = None) -> bool:
    """Checks whether the VMT driver is currently registered in SteamVR."""
    drivers = get_registered_drivers()
    for d in drivers:
        norm_d = os.path.normpath(d).lower()
        if vmt_dir and norm_d == os.path.normpath(vmt_dir).lower():
            return True
        if "vmt" in norm_d or "virtualmotiontracker" in norm_d:
            manifest = Path(d) / "driver.vrdrivermanifest"
            if manifest.exists():
                return True
    return False


def register_driver_in_steamvr(driver_path: str) -> Tuple[bool, str]:
    """
    Registers an external driver folder with SteamVR.
    Uses vrpathreg.exe if available, falling back to direct openvrpaths.vrpath edit.
    """
    abs_driver = os.path.abspath(driver_path)
    manifest = Path(abs_driver) / "driver.vrdrivermanifest"
    if not manifest.exists():
        return False, f"Missing driver.vrdrivermanifest in {abs_driver}"

    # Try vrpathreg.exe first
    vrpathreg = find_vrpathreg_binary()
    if vrpathreg:
        try:
            cmd = [str(vrpathreg), "adddriver", abs_driver]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                return True, f"Successfully registered via vrpathreg: {abs_driver}"
        except Exception as e:
            print(f"[VMT Setup] vrpathreg failed: {e}, falling back to direct JSON edit")

    # Fallback to direct openvrpaths.vrpath edit
    vrpaths = read_openvrpaths()
    if vrpaths is None:
        vrpaths = {
            "config": [],
            "external_drivers": [],
            "jsonid": "vrpathreg",
            "log": [],
            "runtime": [],
            "version": 1,
        }

    ext_drivers = vrpaths.get("external_drivers", [])
    normalized_list = [os.path.normpath(x).lower() for x in ext_drivers]
    if os.path.normpath(abs_driver).lower() not in normalized_list:
        ext_drivers.append(abs_driver)
        vrpaths["external_drivers"] = ext_drivers
        if write_openvrpaths(vrpaths):
            return True, f"Successfully added {abs_driver} to openvrpaths.vrpath"
        return False, "Failed to write openvrpaths.vrpath"
    else:
        return True, f"Driver already registered: {abs_driver}"


def unregister_driver_in_steamvr(driver_path: str) -> Tuple[bool, str]:
    """Unregisters an external driver folder from SteamVR."""
    abs_driver = os.path.abspath(driver_path)
    vrpathreg = find_vrpathreg_binary()
    if vrpathreg:
        try:
            cmd = [str(vrpathreg), "removedriver", abs_driver]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                return True, f"Removed driver via vrpathreg: {abs_driver}"
        except Exception:
            pass

    vrpaths = read_openvrpaths()
    if vrpaths and "external_drivers" in vrpaths:
        target_norm = os.path.normpath(abs_driver).lower()
        new_list = [x for x in vrpaths["external_drivers"] if os.path.normpath(x).lower() != target_norm]
        vrpaths["external_drivers"] = new_list
        if write_openvrpaths(vrpaths):
            return True, f"Removed {abs_driver} from openvrpaths.vrpath"
    return False, "Driver not found or openvrpaths unreadable"


def check_vmt_driver_status() -> Dict[str, Any]:
    """Returns a full diagnostic status dictionary of SteamVR and VMT driver readiness."""
    vrpathreg = find_vrpathreg_binary()
    openvrpaths = read_openvrpaths()
    drivers = get_registered_drivers()
    registered = is_vmt_registered()

    return {
        "steamvr_installed": vrpathreg is not None or (openvrpaths is not None and bool(openvrpaths.get("runtime"))),
        "vrpathreg_path": str(vrpathreg) if vrpathreg else None,
        "openvrpaths_found": openvrpaths is not None,
        "openvrpaths_path": str(get_openvrpaths_path()),
        "registered_external_drivers": drivers,
        "vmt_registered": registered,
        "default_vmt_port": 39570,
    }


def print_vmt_status():
    """Prints diagnostic report to console."""
    status = check_vmt_driver_status()
    print("=================================================================")
    print("  STEAMVR & VIRTUAL MOTION TRACKER (VMT) STATUS REPORT")
    print("=================================================================")
    print(f"  SteamVR Installed       : {'YES' if status['steamvr_installed'] else 'NO / Custom Path'}")
    print(f"  vrpathreg.exe Found     : {status['vrpathreg_path'] or 'Not in standard location'}")
    print(f"  openvrpaths.vrpath      : {status['openvrpaths_path']} ({'Found' if status['openvrpaths_found'] else 'Not created yet'})")
    print(f"  VMT Driver Registered   : {'YES (Ready for SteamVR)' if status['vmt_registered'] else 'NO (Run setup to register)'}")
    print(f"  Registered Drivers ({len(status['registered_external_drivers'])}):")
    for d in status['registered_external_drivers']:
        print(f"    - {d}")
    print(f"  Target UDP Port         : {status['default_vmt_port']}")
    print("=================================================================\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "register" and len(sys.argv) > 2:
        ok, msg = register_driver_in_steamvr(sys.argv[2])
        print(msg)
    elif len(sys.argv) > 1 and sys.argv[1] == "unregister" and len(sys.argv) > 2:
        ok, msg = unregister_driver_in_steamvr(sys.argv[2])
        print(msg)
    else:
        print_vmt_status()
