"""
setup_vmt.py - Quick setup and diagnostic utility for SteamVR Virtual Motion Tracker (VMT).
Run this script to check SteamVR driver status or register your VMT driver with SteamVR.

Usage:
  python setup_vmt.py               # Check status and show setup guide
  python setup_vmt.py --register "C:\\path\\to\\vmt"   # Register custom VMT driver path
  python setup_vmt.py --unregister "C:\\path\\to\\vmt" # Unregister VMT driver path
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from vr.vmt_driver import (
    check_vmt_driver_status,
    print_vmt_status,
    register_driver_in_steamvr,
    unregister_driver_in_steamvr,
)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--register":
        path = sys.argv[2]
        ok, msg = register_driver_in_steamvr(path)
        print(f"[VMT Setup] {msg}")
        return

    if len(sys.argv) > 2 and sys.argv[1] == "--unregister":
        path = sys.argv[2]
        ok, msg = unregister_driver_in_steamvr(path)
        print(f"[VMT Setup] {msg}")
        return

    print_vmt_status()

    status = check_vmt_driver_status()
    if not status["vmt_registered"]:
        print("HOW TO INSTALL & ENABLE VMT IN STEAMVR:")
        print("1. Download the Virtual Motion Tracker (VMT) release from:")
        print("   https://github.com/gpsnmeajp/VirtualMotionTracker/releases")
        print("2. Run VirtualMotionTrackerSetup_0.15.exe, OR extract the VMT driver folder.")
        print("3. Register the VMT driver path with SteamVR by running:")
        print('   python setup_vmt.py --register "C:\\Path\\To\\VMT"\n')
        print("4. Restart SteamVR. Virtual controllers will now spawn automatically")
        print("   when you run GestureDrive in Advanced VR Mode ([V]).\n")
    else:
        print(">>> VMT is registered! Launch SteamVR, then run:")
        print("    python app/main.py")
        print("    Press [V] to switch to Advanced VR Mode and [T] to stream to SteamVR.\n")


if __name__ == "__main__":
    main()
