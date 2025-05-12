#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# IB++ Package Manager 
import os
import sys
import shutil
import urllib.request
import json

# Where packages will be installed
PKG_DIR = os.path.expanduser("~/.ibpp-packages")

# Example: central repo URL (could be a GitHub raw folder or your own server)
REPO_URL = "https://raw.githubusercontent.com/maelruiz/IBplusplus/packages/main/"

def ensure_pkg_dir():
    if not os.path.exists(PKG_DIR):
        os.makedirs(PKG_DIR)

def install(pkg_name):
    ensure_pkg_dir()
    manifest_url = REPO_URL + pkg_name + "/ibpp.json"
    try:
        with urllib.request.urlopen(manifest_url) as resp:
            manifest = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Could not fetch manifest for '{pkg_name}': {e}")
        return

    entry_file = manifest.get("entry")
    if not entry_file:
        print("Invalid manifest: no entry file specified.")
        return

    # Download all files listed in manifest (expand as needed)
    files = manifest.get("files", [entry_file])
    pkg_path = os.path.join(PKG_DIR, pkg_name)
    os.makedirs(pkg_path, exist_ok=True)
    for f in files:
        file_url = REPO_URL + pkg_name + "/" + f
        dest = os.path.join(pkg_path, f)
        try:
            with urllib.request.urlopen(file_url) as resp, open(dest, "wb") as out:
                out.write(resp.read())
            print(f"Downloaded {f}")
        except Exception as e:
            print(f"Failed to download {f}: {e}")

    # Save manifest
    with open(os.path.join(pkg_path, "ibpp.json"), "w") as out:
        json.dump(manifest, out)
    print(f"Installed {pkg_name} to {pkg_path}")

def list_pkgs():
    ensure_pkg_dir()
    pkgs = os.listdir(PKG_DIR)
    if not pkgs:
        print("No packages installed.")
        return
    print("Installed packages:")
    for p in pkgs:
        print(" -", p)

def remove(pkg_name):
    pkg_path = os.path.join(PKG_DIR, pkg_name)
    if os.path.exists(pkg_path):
        shutil.rmtree(pkg_path)
        print(f"Removed {pkg_name}")
    else:
        print(f"Package {pkg_name} not found.")

def help():
    print("IB++ Package Manager")
    print("Usage:")
    print("  ibppkg.py install <package>")
    print("  ibppkg.py list")
    print("  ibppkg.py remove <package>")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        help()
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "install" and len(sys.argv) == 3:
        install(sys.argv[2])
    elif cmd == "list":
        list_pkgs()
    elif cmd == "remove" and len(sys.argv) == 3:
        remove(sys.argv[2])
    else:
        help()