# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc
#
# This file was produced as part of a research project conducted for
# The World Bank Group and is licensed under the terms of the MIT license.
#
# For inquiries, contact: info@geostack.ca
# Repository: https://github.com/geo-stack/sahel_water_table_ml
# =============================================================================

from pathlib import Path
import nbformat
from nbconvert import PythonExporter

file = Path(__file__).parent
filenames = [
    file / "Notebook_meteo_features.ipynb",
    file / "Notebook_model.ipynb",
    file / "Notebook_topo_features.ipynb",
    file / "script_topo_features.ipynb"
    ]
for notebook_path in filenames:
    print(f'Convertion {notebook_path.name}...')

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Convert to Python script
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(notebook)

    # Determine output path
    output_path = notebook_path.with_suffix('.py')

    # Write script
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script)
