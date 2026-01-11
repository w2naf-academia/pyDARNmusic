# pyDARNmusic

**Multiple Signal Classification (MUSIC) Algorithm for SuperDARN MSTID Detection**

A Python library implementing the MUSIC algorithm for detecting and characterizing Medium-Scale Traveling Ionospheric Disturbances (MSTIDs) using SuperDARN radar data. This package provides the core signal processing capabilities that power the [DARNtids](https://github.com/w2naf-academia/DARNtids) analysis pipeline.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)](LICENSE)

## Project History

pyDARNmusic was originally developed by **Dr. Nathaniel A. Frissell** as part of the DaViTpy library to support MSTID analysis published in:

> Frissell, N. A., J. B. H. Baker, J. M. Ruohoniemi, A. J. Gerrard, E. S. Miller, J. P. Marini, M. L. West, and W. A. Bristow (2014), Climatology of medium-scale traveling ionospheric disturbances observed by the midlatitude Blackstone SuperDARN radar, *J. Geophys. Res. Space Physics*, 119, doi:[10.1002/2014JA019870](https://doi.org/10.1002/2014JA019870).

This work demonstrated the application of the MUSIC (Multiple Signal Classification) algorithm to SuperDARN ground scatter data for detecting and characterizing MSTIDs as manifestations of atmospheric gravity waves.

### Python 3 Migration and Modernization (2023)

**Francis Hassan Tholley** completely reimplemented the MUSIC algorithm for Python 3 compatibility and modern SuperDARN data libraries. This migration involved:

- **Python 2 to 3 Transition**: Complete rewrite for Python 3.8+ compatibility
- **Library Migration**: Replaced deprecated DaViTpy with actively developed PyDARN/pyDARNio libraries
- **Performance Optimization**: Rewrote nested loops using NumPy vectorized operations, dramatically reducing runtime
- **Architecture Modernization**: Applied software engineering best practices including single responsibility and open/closed principles
- **Enhanced Plotting**: Updated visualization using Cartopy for modern geospatial plotting

This work was completed as part of his MS Software Engineering thesis:

> Tholley, F. H. (2023), PyDARNMUSIC/PyDARNMUSICWeb: Software for SuperDARN Medium Scale Traveling Ionospheric Disturbance Visualization and Analysis, *MS Thesis, University of Scranton*, [https://archives.scranton.edu/digital/collection/p15111coll1/id/1403/rec/1](https://archives.scranton.edu/digital/collection/p15111coll1/id/1403/rec/1).

## Overview

pyDARNmusic implements the MUSIC (Multiple Signal Classification) cross-spectral analysis algorithm for SuperDARN radar data. The algorithm analyzes ground scatter patterns to detect wave-like structures in the ionosphere and extract their properties.

### What is MUSIC?

MUSIC is a signal processing technique that:
- Decomposes radar backscatter data into wavenumber-frequency space
- Identifies multiple simultaneous wave components
- Extracts wave parameters: wavelength, period, velocity, propagation azimuth
- Works across all radar beams simultaneously for robust detection

Unlike single-beam range-time-intensity (RTI) analysis, MUSIC uses the full radar field-of-view to overcome instrumental biases and detect waves at any orientation.

### Key Features

- **Data Loading**: Automated ingestion of SuperDARN fitacf files with flexible date range selection
- **MUSIC Processing**: Complete implementation of the MUSIC algorithm following Samson et al. [1990]
  - Time and spatial interpolation for uniform grids
  - Bandpass filtering for MSTID frequency isolation
  - FFT and cross-spectral matrix computation
  - Horizontal wavenumber array (K-array) calculation
  - Multi-signal detection with watershed algorithm
- **Multiple Scatter Models**: Ground scatter (GS), ionospheric scatter (IS), and various elevation models
- **Coordinate Systems**: Geographic coordinates (tested), with support for magnetic coordinates
- **Visualization Tools**:
  - Range-Time-Parameter (RTP) plots
  - Fan plots showing spatial structure
  - K-array spectrograms
  - Multi-panel time series and spectral analysis
- **High Performance**: NumPy-optimized array operations for fast processing

## Scientific Background

Medium-Scale Traveling Ionospheric Disturbances (MSTIDs) are quasi-periodic perturbations of the F-region ionosphere with:
- **Periods**: 15-60 minutes
- **Wavelengths**: 50-500 km (typically 100-300 km)
- **Velocities**: 50-250 m/s
- **Sources**: Atmospheric gravity waves from auroral/geomagnetic activity, tropospheric convection, and other atmospheric processes

SuperDARN radars observe MSTIDs as wave-like patterns in ground scatter. Ionospheric density perturbations create focusing/defocusing of HF radar signals, producing alternating bands of enhanced and reduced backscatter power that move with the disturbance.

## Installation

### Prerequisites

- Python >= 3.8
- SuperDARN fitacf data files

### Option 1: Using Conda/Mamba (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/w2naf-academia/pyDARNmusic.git
cd pyDARNmusic
```

2. Create and activate a conda environment:
```bash
conda create -n pyDARNmusic python=3.8
conda activate pyDARNmusic
```

3. Install dependencies and package:
```bash
pip install -r requirements.txt
pip install -e .
```

### Option 2: Using pip only

1. Clone the repository:
```bash
git clone https://github.com/w2naf-academia/pyDARNmusic.git
cd pyDARNmusic
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Basic MUSIC Analysis

```python
import datetime
from pyDARNmusic import load_fitacf
from pyDARNmusic.music import musicArray

# Define parameters
radar = 'bks'  # Blackstone radar
sTime = datetime.datetime(2010, 11, 19, 14, 0)
eTime = datetime.datetime(2010, 11, 19, 16, 0)
data_dir = '/path/to/superdarn/data'

# Load fitacf data
fitacf = load_fitacf(radar, sTime, eTime, data_dir=data_dir)

# Create MUSIC array object
dataObj = musicArray(fitacf, sTime=sTime, eTime=eTime,
                     fovModel='GS',  # Ground scatter model
                     param='p_l')    # Power parameter

# Process data
from pyDARNmusic import (defineLimits, applyLimits, timeInterpolation,
                         beamInterpolation, filterTimes, detrend,
                         windowData, calculateFFT, calculateDlm,
                         calculateKarr, detectSignals)

# Define time and range limits
dataObj = defineLimits(dataObj, rangeLimits=(200, 600),
                       timeLimits=(sTime, eTime))

# Apply limits and interpolate
dataObj = applyLimits(dataObj)
dataObj = timeInterpolation(dataObj)
dataObj = beamInterpolation(dataObj)

# Filter and process
dataObj = filterTimes(dataObj, numTaps=101,
                      cutoff_low=0.5e-3, cutoff_high=1.2e-3)
dataObj = detrend(dataObj)
dataObj = windowData(dataObj)

# Compute MUSIC arrays
dataObj = calculateFFT(dataObj)
dataObj = calculateDlm(dataObj)
dataObj = calculateKarr(dataObj)

# Detect signals
dataObj = detectSignals(dataObj)

# Plot results
from pyDARNmusic.plotting import plotKarrDetected
plotKarrDetected(dataObj)
```

### Visualization

```python
from pyDARNmusic.plotting import musicRTP, musicFan

# Range-Time-Parameter plot for specific beam
fig = musicRTP(dataObj, beam=7)

# Fan plot at specific time
plotTime = datetime.datetime(2010, 11, 19, 14, 30)
fig = musicFan(dataObj, time=plotTime)
```

## Package Structure

```
pyDARNmusic/
├── pyDARNmusic/           # Main package
│   ├── io/                # Data loading
│   │   └── load_fitacf.py # SuperDARN fitacf file reader
│   ├── music/             # Core MUSIC algorithm
│   │   ├── music_array.py      # Main container class
│   │   ├── music_data_object.py # Data object handling
│   │   ├── signal_filter.py    # Signal filtering
│   │   └── signals_detected.py # Detection results
│   ├── plotting/          # Visualization
│   │   ├── musicPlot.py   # Core plotting functions
│   │   ├── rtp.py         # Range-Time-Parameter plots
│   │   └── fan.py         # Fan plots
│   └── utils/             # Utilities
│       ├── musicUtils.py  # MUSIC processing functions
│       ├── timeUtils.py   # Time handling utilities
│       ├── geoPack.py     # Geographic calculations
│       ├── plotUtils.py   # Plotting helpers
│       ├── radUtils.py    # Radar utilities
│       └── rtpUtils.py    # RTP utilities
├── scripts/               # Example scripts
├── notebooks/             # Jupyter notebooks with examples
├── TestData/              # Test datasets
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Core Components

### musicArray

The main container class for MUSIC data processing. Handles:
- SuperDARN data ingestion from fitacf records
- Field-of-view (FOV) calculation with multiple scatter models
- Beam and range gate organization
- Ground scatter filtering
- Data quality checks

### musicDataObj

Data object class for processed datasets with:
- Time series management
- Multi-dimensional data arrays (time x beam x range)
- Metadata handling
- Processing history tracking
- Copy and transformation methods

### Processing Functions

Core MUSIC algorithm steps available in `musicUtils`:
- `defineLimits()` - Set time and range boundaries
- `applyLimits()` - Apply spatial/temporal limits
- `timeInterpolation()` - Uniform time sampling
- `beamInterpolation()` - Spatial interpolation across beams
- `filterTimes()` - Bandpass filtering for MSTID frequencies
- `detrend()` - Remove linear trends
- `windowData()` - Apply windowing function
- `calculateFFT()` - Fast Fourier Transform
- `calculateDlm()` - Cross-spectral matrix
- `calculateKarr()` - Horizontal wavenumber array
- `detectSignals()` - Identify wave components

## Scatter Mapping Models

pyDARNmusic supports multiple field-of-view models via the `fovModel` parameter:

- `GS` - Ground Scatter model (Bristow et al. 1994) - Default for MSTID studies
- `IS` - Standard SuperDARN ionospheric scatter model
- `S` - Standard projection model
- `E1` - Chisham E-region 1/2-hop model
- `F1` - Chisham F-region 1/2-hop model
- `F3` - Chisham F-region 1.5-hop model
- `C` - Chisham projection model
- `None` - Use provided elevation/altitude values directly

## Dependencies

### Core Scientific
- `numpy` - Array operations and numerical computing
- `scipy` - Signal processing (via scikit-image)
- `matplotlib` - Visualization
- `cartopy` - Geospatial plotting
- `scikit-image` - Watershed algorithm for signal detection

### SuperDARN Ecosystem
- `pydarn` - SuperDARN data utilities and radar information
- `pydarnio` - SuperDARN file I/O
- `aacgmv2` - Magnetic coordinate transformations

### Geospatial
- `pygeos` / `geos` - Geometric operations
- `pycurl` - Data downloading

### Utilities
- `tqdm` - Progress bars

## Usage in DARNtids

pyDARNmusic serves as the core MSTID detection engine for the [DARNtids](https://github.com/w2naf-academia/DARNtids) analysis pipeline. DARNtids extends pyDARNmusic with:
- Automated multi-event processing
- MSTID event classification
- Database management (MongoDB)
- Calendar visualizations
- Multi-radar network analysis
- HDF5 data storage

See the DARNtids repository for large-scale MSTID analysis workflows.

## Algorithm Reference

The MUSIC implementation follows:

> Samson, J. C., R. A. Greenwald, J. M. Ruohoniemi, T. J. Hughes, and D. D. Wallis (1990), Magnetometer and radar observations of magnetohydrodynamic cavity modes in the Earth's magnetosphere, *Can. J. Phys.*, 68, 1265-1290.

> Bristow, W. A., R. A. Greenwald, and J. C. Samson (1994), Identification of high-latitude acoustic gravity wave sources using the Goose Bay HF Radar, *J. Geophys. Res.*, 99(A1), 319-331, doi:[10.1029/93JA01470](https://doi.org/10.1029/93JA01470).

> Schmidt, R. O. (1986), Multiple emitter location and signal parameter estimation, *IEEE Trans. Antennas Propag.*, AP-34(3), 276-280.

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use pyDARNmusic in your research, please cite:

**Original Algorithm Application:**
```bibtex
@article{frissell2014climatology,
  author = {Frissell, N. A. and Baker, J. B. H. and Ruohoniemi, J. M. and Gerrard, A. J. and Miller, E. S. and Marini, J. P. and West, M. L. and Bristow, W. A.},
  title = {Climatology of medium-scale traveling ionospheric disturbances observed by the midlatitude Blackstone SuperDARN radar},
  journal = {Journal of Geophysical Research: Space Physics},
  volume = {119},
  year = {2014},
  doi = {10.1002/2014JA019870}
}
```

**Software:**
```bibtex
@software{pydarnmusic,
  author = {Frissell, Nathaniel A. and Tholley, Francis Hassan},
  title = {pyDARNmusic: Multiple Signal Classification Algorithm for SuperDARN MSTID Detection},
  url = {https://github.com/w2naf-academia/pyDARNmusic},
  year = {2023}
}
```

**Python 3 Migration:**
```bibtex
@mastersthesis{tholley2023pydarnmusic,
  author = {Tholley, Francis Hassan},
  title = {PyDARNMUSIC/PyDARNMUSICWeb: Software for SuperDARN Medium Scale Traveling Ionospheric Disturbance Visualization and Analysis},
  school = {University of Scranton},
  year = {2023},
  type = {MS Thesis},
  url = {https://archives.scranton.edu/digital/collection/p15111coll1/id/1403/rec/1}
}
```

**MUSIC Algorithm:**
```bibtex
@article{bristow1994identification,
  author = {Bristow, W. A. and Greenwald, R. A. and Samson, J. C.},
  title = {Identification of high-latitude acoustic gravity wave sources using the Goose Bay HF Radar},
  journal = {Journal of Geophysical Research},
  volume = {99},
  number = {A1},
  pages = {319--331},
  year = {1994},
  doi = {10.1029/93JA01470}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE](LICENSE) file for details.

## Authors and Contributors

**Principal Investigator:**
- **Nathaniel A. Frissell, PhD** - Original developer and project lead
  - Email: nathaniel.frissell@scranton.edu
  - Website: https://hamsci.org

**Major Contributors:**
- **Francis Hassan Tholley, MS** - Python 3 reimplementation and modernization (2023)
  - Complete algorithm rewrite for Python 3.8+
  - Performance optimization with NumPy vectorization
  - Migration from DaViTpy to PyDARN ecosystem
  - Enhanced visualization with Cartopy

## Acknowledgments

### Funding Support

This work has been supported by:
- NASA Grant 80NSSC21K0002
- NSF Grant AGS-0838219

### Community and Infrastructure

- SuperDARN community for radar data and infrastructure
- PyDARN development team for data access libraries
- HamSCI (Ham Radio Science Citizen Initiative)
- University of Scranton Department of Computing Sciences
- Claude (Anthropic) for documentation assistance

## Support

- **Issues**: [GitHub Issues](https://github.com/w2naf-academia/pyDARNmusic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/w2naf-academia/pyDARNmusic/discussions)
- **Email**: nathaniel.frissell@scranton.edu

## Related Projects

- [DARNtids](https://github.com/w2naf-academia/DARNtids) - Full MSTID analysis pipeline using pyDARNmusic
- [pydarn](https://github.com/SuperDARN/pydarn) - Python library for SuperDARN data
- [pydarnio](https://github.com/SuperDARN/pyDARNio) - SuperDARN file I/O
- [SuperDARN](http://vt.superdarn.org/) - Global HF radar network
