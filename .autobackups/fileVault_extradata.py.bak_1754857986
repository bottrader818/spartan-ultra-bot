#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
import json
import plistlib
import subprocess
import sys

# Constants
kCoreStorageLogicalVolumeConversionState = 'CoreStorageLogicalVolumeConversionState'
kCoreStorageLogicalVolumeFamilyEncryptionType = 'CoreStorageLogicalVolumeFamilyEncryptionType'
kCoreStorageLogicalVolumeSparse = 'CoreStorageLogicalVolumeGroupSparse'
kMemberOfCoreStorageLogicalVolumeFamily = 'MemberOfCoreStorageLogicalVolumeFamily'
kMemberOfCoreStorageLogicalVolumeGroup = 'MemberOfCoreStorageLogicalVolumeGroup'


# Implementation
def _diskutil_cs_info(plist=False, target='/'):
    cmd = ['diskutil', 'coreStorage', 'information']
    cmd.append('-plist') if plist else None
    cmd.append(target) if isinstance(target, (str, unicode)) else None
    try:
        ret = subprocess.check_output(cmd)
    except Exception:
        return {}

    return plistlib.readPlistFromString(ret) if plist else str(ret)


def _diskutil_cs_list(plist=False):
    cmd = ['diskutil', 'coreStorage', 'list']
    cmd.append('-plist') if plist else None
    try:
        ret = subprocess.check_output(cmd)
    except Exception:
        return {}

    return plistlib.readPlistFromString(ret) if plist else str(ret)


def _diskutil_list(plist=False):
    cmd = ['diskutil', 'list']
    cmd.append('-plist') if plist else None
    try:
        ret = subprocess.check_output(cmd)
    except Exception:
        return {}

    return plistlib.readPlistFromString(ret) if plist else str(ret)


# Go!
if __name__ == "__main__":

    try:
        data = {
            'diskutil_cs_info': _diskutil_cs_info(target='/'),
            'diskutil_cs_info_plist': _diskutil_cs_info(plist=True, target='/'),
            'diskutil_cs_list': _diskutil_cs_list(),
            'diskutil_cs_list_plist': _diskutil_cs_list(plist=True),
            'diskutil_list': _diskutil_list(),
            'diskutil_list_plist': _diskutil_list(plist=True),
        }

        out = json.dumps(data)
    except Exception:

        sys.stdout.write(json.dumps({}))
        sys.stdout.flush()
        sys.exit(1)
    else:
        sys.stdout.write(out)
        sys.stdout.flush()
        sys.exit(0)
