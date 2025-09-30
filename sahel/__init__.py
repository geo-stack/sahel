# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel
# =============================================================================

from appconfigs.user import UserConfig
import os.path as osp

__datadir__ = osp.join(osp.dirname(osp.dirname(__file__)), 'data')

# =============================================================================
# Config instance
# =============================================================================
# IMPORTANT NOTES:
# 1. If you want to *change* the default value of a current option, you need to
#    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
# 2. If you want to *remove* options that are no longer needed in our codebase,
#    or if you want to *rename* options, then you need to do a MAJOR update in
#    version, e.g. from 3.0.0 to 4.0.0
# 3. You don't need to touch this value if you're just adding a new option
CONF_VERSION = '0.0.1'

APPNAME = 'sahel'
CONFIG_DIR = osp.join(osp.dirname(osp.dirname(__file__)), '.configs')
DEFAULTS = []

# Setup the main configuration instance.
try:
    CONF = UserConfig(APPNAME, defaults=DEFAULTS, load=True,
                      version=CONF_VERSION, path=CONFIG_DIR,
                      backup=True, raw_mode=True)
except Exception:
    CONF = UserConfig(APPNAME, defaults=DEFAULTS, load=False,
                      version=CONF_VERSION, path=CONFIG_DIR,
                      backup=True, raw_mode=True)
