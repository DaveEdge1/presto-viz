# Presto Visualization

**Automated paleoclimate reconstruction visualization pipeline**

Generate interactive HTML visualizations from paleoclimate reconstruction NetCDF files. This tool processes output from climate reconstruction methods (like DA Holocene and GraphEM) and creates comprehensive visual summaries including maps, time series, and ensemble statistics.

[![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-blue)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 📊 **Automated Processing**: Three-stage pipeline from raw NetCDF to interactive HTML
- 🗺️ **Spatial Visualization**: Generate maps of temperature anomalies and trends
- 📈 **Time Series**: Create time series plots with ensemble uncertainties
- 🌐 **Interactive HTML**: Self-contained visualizations viewable in any browser
- ⚡ **GitHub Actions Integration**: Run directly from your workflow
- 🐍 **Python-based**: Built with xarray, numpy, and standard scientific libraries

## Quick Start

### Using with GitHub Actions (Recommended)

Add to your workflow after generating NetCDF outputs:

```yaml
jobs:
  your-reconstruction:
    # ... your reconstruction steps ...
    - name: Upload NetCDF outputs
      uses: actions/upload-artifact@v4
      with:
        name: reconstruction-output
        path: output/*.nc

  visualize:
    needs: your-reconstruction
    uses: DaveEdge1/presto-viz/.github/workflows/presto-viz-reusable.yml@main
    with:
      reconstruction_id: ${{ github.run_number }}
      artifact_name: reconstruction-output
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/DaveEdge1/presto-viz.git
cd presto-viz

# Create conda environment
conda env create -f presto_env.yml
conda activate presto_env
pip install psutil

# Run the pipeline
python 1_format_data_daholocene_graphem.py /path/to/input/
python 2_make_maps_and_ts.py /path/to/input/ /path/to/output/
python 3_make_html_file.py /path/to/input/ /path/to/output/ ./web_assets/
```

## Pipeline Overview

The visualization pipeline consists of three sequential scripts:

### 1. Format Data (`1_format_data_daholocene_graphem.py`)
- **Input**: Raw reconstruction NetCDF files
- **Output**: Standardized NetCDF format
- **Purpose**: Normalizes data structure for consistent processing

### 2. Generate Maps and Time Series (`2_make_maps_and_ts.py`)
- **Input**: Standardized NetCDF from step 1
- **Output**: PNG images of maps and time series
- **Purpose**: Creates all visualization components

### 3. Create HTML (`3_make_html_file.py`)
- **Input**: Images from step 2 + web assets
- **Output**: Interactive HTML file
- **Purpose**: Packages everything into a single viewable file

## Input Requirements

Your NetCDF file should contain:

- **Spatial data**: Temperature fields with dimensions `(time, lat, lon)` or `(age, lat, lon)`
- **Global means**: Spatially-averaged temperature time series
- **Ensemble members**: Multiple realizations (optional but recommended)
- **Coordinates**: `lat`, `lon`, `time` or `age`

Example structure:
```python
<xarray.Dataset>
Dimensions:     (age: 1000, lat: 72, lon: 144, ens: 100)
Coordinates:
  * age         (age) float64 0.0 10.0 20.0 ... 9990.0
  * lat         (lat) float64 -88.75 -86.25 ... 88.75
  * lon         (lon) float64 -178.75 -176.25 ... 178.75
  * ens         (ens) int64 1 2 3 ... 100
Data variables:
    recon_tas_mean       (age, lat, lon) float64
    recon_tas_ens        (ens, age, lat, lon) float64
    recon_tas_global_mean (ens, age) float64
```

## Configuration

### Supported Reconstruction Methods

- **DA Holocene**: Holocene data assimilation reconstructions
- **GraphEM**: Graphical Expectation-Maximization reconstructions

### Directory Structure

```
your-project/
├── data/
│   ├── reconstruction_id/
│   │   ├── holocene_recon*.nc  # or graph_em output
│   │   └── configs.yml
└── output/
    └── reconstruction_id/
        ├── *.png              # Generated visualizations
        ├── *.html             # Interactive viewer
        └── *.log              # Processing logs
```

## GitHub Actions Integration

### Full Example for LMR2 Repository

```yaml
name: Climate Reconstruction with Visualization

on:
  workflow_dispatch:
  push:
    branches: [main]

jobs:
  run-reconstruction:
    name: Run Climate Reconstruction
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Run reconstruction
        run: |
          python run_cfr_analysis.py --config config.yml

      - name: Upload reconstruction outputs
        uses: actions/upload-artifact@v4
        with:
          name: cfr-netcdf-output
          path: |
            output/**/*.nc
            output/**/configs.yml

  create-visualization:
    name: Generate Visualization
    needs: run-reconstruction
    uses: DaveEdge1/presto-viz/.github/workflows/presto-viz-reusable.yml@main
    with:
      reconstruction_id: Run_${{ github.run_number }}
      artifact_name: cfr-netcdf-output
```

### Workflow Inputs

| Input | Required | Description | Default |
|-------|----------|-------------|---------|
| `reconstruction_id` | Yes | Unique identifier for this run | - |
| `artifact_name` | Yes | Name of artifact containing NetCDF | - |
| `data_dir` | No | Custom data directory path | `./data/{id}` |
| `output_dir` | No | Custom output directory path | `./output/{id}` |
| `web_data_dir` | No | Web assets directory | `./web_assets` |

### Workflow Outputs

| Output | Description |
|--------|-------------|
| `visualization_artifact` | Name of the generated visualization artifact |

Artifacts are uploaded as:
- `presto-viz-logs-{reconstruction_id}`: All log files
- `presto-viz-output-{reconstruction_id}`: Visualizations and HTML

## Output Files

After successful completion:

- **`{dataset}_v{version}_tas_annual.nc`**: Standardized NetCDF
- **`*.png`**: Individual visualization images
- **`visualizer_{reconstruction_id}.html`**: Interactive HTML viewer
- **`*.log`**: Processing logs for debugging

## Dependencies

Core dependencies (installed via `presto_env.yml`):
- Python 3.12
- xarray
- numpy
- netCDF4
- matplotlib
- cartopy
- pyyaml
- psutil

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: No such file or directory: 'holocene_recon*.nc'`
- **Solution**: Ensure your input directory contains the expected NetCDF files

**Issue**: `KeyError: 'recon_tas_mean'`
- **Solution**: Verify your NetCDF has the expected variable names

**Issue**: Timeout in GitHub Actions
- **Solution**: Workflow has 3-hour timeout. For large datasets, consider:
  - Reducing spatial resolution
  - Limiting ensemble members
  - Splitting into multiple jobs

### Debug Mode

Run with verbose logging:
```bash
python -u 1_format_data_daholocene_graphem.py /path/to/data/ 2>&1 | tee format.log
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this software in your research, please cite:

```bibtex
@software{presto_viz,
  author = {Erb, Michael and {PaleoPresto Team}},
  title = {Presto Visualization: Automated Paleoclimate Reconstruction Visualization},
  year = {2024},
  url = {https://github.com/DaveEdge1/presto-viz},
  note = {GitHub repository}
}
```

## Related Projects

- **[PaleoPresto](https://paleopresto.com)**: Paleoclimate reconstruction platform
- **[LMR2](https://github.com/DaveEdge1/LMR2)**: Last Millennium Reanalysis version 2

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Originally developed for the PaleoPresto platform
- Built with support from [your funding sources]
- Thanks to the paleoclimate community for feedback and testing

## Contact

- **Issues**: [GitHub Issues](https://github.com/DaveEdge1/presto-viz/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DaveEdge1/presto-viz/discussions)
- **PaleoPresto**: https://paleopresto.com

---

**Note**: This repository contains only the visualization component. For the complete PaleoPresto reconstruction platform, see the main [prestoServer](https://github.com/DaveEdge1/prestoServer) repository (private).
