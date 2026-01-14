# pyDARNmusic

**Multiple Signal Classification (MUSIC) Algorithm for SuperDARN MSTID Detection**

A Python library implementing the MUSIC algorithm for detecting and characterizing Medium-Scale Traveling Ionospheric Disturbances (MSTIDs) using SuperDARN radar data. This package enables **single-event analysis** of MSTID wave properties and provides the core signal processing capabilities that power the [DARNtids](https://github.com/w2naf-academia/DARNtids) multi-event analysis pipeline.

[![PyPI version](https://badge.fury.io/py/pyDARNmusic.svg)](https://badge.fury.io/py/pyDARNmusic)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)](LICENSE)

---

## Table of Contents

- [Project History](#project-history)
- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [Core Components](#core-components)
- [Scatter Mapping Models](#scatter-mapping-models)
- [Dependencies](#dependencies)
- [Relationship to DARNtids](#relationship-to-darntids)
- [Algorithm References](#algorithm-references)
- [Examples and Tutorials](#examples-and-tutorials)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Authors and Contributors](#authors-and-contributors)
- [Acknowledgments](#acknowledgments)
- [Support](#support)
- [Related Projects](#related-projects)

---

## Project History

pyDARNmusic was originally developed by **Dr. Nathaniel A. Frissell** in 2013 as part of the DaViTpy library to enable MUSIC-based MSTID detection on SuperDARN radar data. The initial implementation supported single-event analysis and was published in:

> Frissell, N. A., J. B. H. Baker, J. M. Ruohoniemi, A. J. Gerrard, E. S. Miller, J. P. Marini, M. L. West, and W. A. Bristow (2014), Climatology of medium-scale traveling ionospheric disturbances observed by the midlatitude Blackstone SuperDARN radar, *J. Geophys. Res. Space Physics*, 119, doi:[10.1002/2014JA019870](https://doi.org/10.1002/2014JA019870).

This foundational work demonstrated the application of the MUSIC (Multiple Signal Classification) algorithm to SuperDARN ground scatter data for detecting and characterizing MSTIDs as manifestations of atmospheric gravity waves. The paper presented a climatology of daytime midlatitude MSTIDs observed by the Blackstone radar, identifying two distinct propagation populations and correlations with geomagnetic activity.

The success of pyDARNmusic's single-event analysis capabilities led to the development of **DARNtids** (2016), which extends pyDARNmusic to enable multi-event statistical studies, climatologies, and automated processing across multiple radars.

### Python 3 Migration and Modernization (2023)

**Francis Hassan Tholley** completely reimplemented the MUSIC algorithm for Python 3 compatibility and modern SuperDARN data libraries. This migration involved:

- **Python 2 to 3 Transition**: Complete rewrite for Python 3.8+ compatibility
- **Library Migration**: Replaced deprecated DaViTpy with actively developed PyDARN/pyDARNio libraries
- **Performance Optimization**: Rewrote nested loops using NumPy vectorized operations, drastically reducing runtime (critical for the multi-event processing in DARNtids)
- **Architecture Modernization**: Applied software engineering best practices including single responsibility and open/closed principles
- **Enhanced Plotting**: Updated visualization using Cartopy for modern geospatial plotting

This work was completed as part of his MS Software Engineering thesis:

> Tholley, F. H. (2023), PyDARNMUSIC/PyDARNMUSICWeb: Software for SuperDARN Medium Scale Traveling Ionospheric Disturbance Visualization and Analysis, *MS Thesis, University of Scranton*, [https://archives.scranton.edu/digital/collection/p15111coll1/id/1403/rec/1](https://archives.scranton.edu/digital/collection/p15111coll1/id/1403/rec/1).

## Overview

pyDARNmusic implements the MUSIC (Multiple Signal Classification) cross-spectral analysis algorithm for SuperDARN radar data. The algorithm analyzes ground scatter patterns to detect wave-like structures in the ionosphere and extract their properties from **individual MSTID events**.

### What is MUSIC?

MUSIC is a signal processing technique that:
- Decomposes radar backscatter data into wavenumber-frequency space
- Identifies multiple simultaneous wave components in a single event
- Extracts wave parameters: wavelength, period, velocity, propagation azimuth
- Works across all radar beams simultaneously for robust detection

Unlike single-beam range-time-intensity (RTI) analysis, MUSIC uses the full radar field-of-view to overcome instrumental biases and detect waves at any orientation.

### Key Features

- **Single-Event Analysis**: Designed for detailed characterization of individual MSTID events
- **Data Loading**: Automated ingestion of SuperDARN fitacf files with flexible date range selection
- **MUSIC Processing**: Complete implementation of the MUSIC algorithm following Samson et al. (1990)
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
  - Access via [SuperDARN Data Portal](http://superdarn.thayer.dartmouth.edu/data.html)
  - Or institutional mirror sites

### Option 1: Install from PyPI (Recommended)

```bash
pip install pyDARNmusic
```

### Option 2: Install from source

1. Clone the repository:
```bash
git clone https://github.com/w2naf/pyDARNmusic.git
cd pyDARNmusic
```

2. Install the package:
```bash
pip install -e .
```

### Option 3: Using Conda environment

1. Clone the repository:
```bash
git clone https://github.com/w2naf/pyDARNmusic.git
cd pyDARNmusic
```

2. Create and activate a conda environment:
```bash
conda create -n pyDARNmusic python=3.11
conda activate pyDARNmusic
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

### Minimal Example

```python
import datetime
from pyDARNmusic import load_fitacf
from pyDARNmusic.music import musicArray

# Load data for a 2-hour event
radar = 'bks'  # Blackstone radar
sTime = datetime.datetime(2010, 11, 19, 14, 0)
eTime = datetime.datetime(2010, 11, 19, 16, 0)
data_dir = '/path/to/superdarn/data'

fitacf = load_fitacf(radar, sTime, eTime, data_dir=data_dir)

# Create MUSIC array object
dataObj = musicArray(fitacf, sTime=sTime, eTime=eTime,
                     fovModel='GS',  # Ground scatter model
                     param='p_l')    # Power parameter

# Generate Range-Time-Intensity plot
from pyDARNmusic import musicRTP
fig = musicRTP(dataObj, beam=7)
```

### Complete MUSIC Analysis Pipeline

```python
import datetime
from pyDARNmusic import (load_fitacf, defineLimits, applyLimits,
                         timeInterpolation, beamInterpolation, filterTimes,
                         detrend, windowData, calculateFFT, calculateDlm,
                         calculateKarr, detectSignals, plotKarrDetected)
from pyDARNmusic.music import musicArray

# Define parameters for event analysis
radar = 'bks'
sTime = datetime.datetime(2010, 11, 19, 14, 0)
eTime = datetime.datetime(2010, 11, 19, 16, 0)
data_dir = '/path/to/superdarn/data'

# Load fitacf data
fitacf = load_fitacf(radar, sTime, eTime, data_dir=data_dir)

# Create MUSIC array object
dataObj = musicArray(fitacf, sTime=sTime, eTime=eTime,
                     fovModel='GS', param='p_l')

# Define spatial and temporal limits
dataObj = defineLimits(dataObj, rangeLimits=(200, 600),
                       timeLimits=(sTime, eTime))

# Apply limits and interpolate to uniform grid
dataObj = applyLimits(dataObj)
dataObj = timeInterpolation(dataObj)
dataObj = beamInterpolation(dataObj)

# Apply bandpass filter for MSTID frequencies (0.5-1.2 mHz = 14-33 min periods)
dataObj = filterTimes(dataObj, numTaps=101,
                      cutoff_low=0.5e-3, cutoff_high=1.2e-3)

# Remove trends and apply window
dataObj = detrend(dataObj)
dataObj = windowData(dataObj)

# Compute MUSIC arrays
dataObj = calculateFFT(dataObj)
dataObj = calculateDlm(dataObj)      # Cross-spectral matrix
dataObj = calculateKarr(dataObj)     # Horizontal wavenumber array

# Detect signals using watershed algorithm
dataObj = detectSignals(dataObj)

# Plot K-array with detected signals
fig = plotKarrDetected(dataObj)
```

### Additional Visualizations

```python
from pyDARNmusic import musicRTP, musicFan

# Range-Time-Parameter plot for specific beam
fig = musicRTP(dataObj, beam=7)

# Fan plot showing spatial structure at specific time
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
│   └── music_script.py    # Working example
├── notebooks/             # Jupyter notebooks with examples
│   ├── MUSIC.ipynb        # Basic MUSIC tutorial
│   ├── MUSIC-Simulate.ipynb     # Simulation examples
│   └── MUSICWORKING-BPK.ipynb   # Buckland Park example
├── TestData/              # Test datasets
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Core Components

### musicArray

The main container class for MUSIC data processing ([music_array.py](pyDARNmusic/music/music_array.py)). Handles:
- SuperDARN data ingestion from fitacf records
- Field-of-view (FOV) calculation with multiple scatter models
- Beam and range gate organization
- Ground scatter filtering (gscat parameter: 0=all, 1=ground only, 2=ionospheric only)
- Data quality checks

**Key Parameters:**
- `fitacf` - Fitacf data records from `load_fitacf()`
- `sTime`, `eTime` - Start and end times
- `param` - Radar parameter ('p_l' for power, 'v' for velocity, etc.)
- `gscat` - Ground scatter filter flag
- `fovModel` - Scatter mapping model (see [Scatter Mapping Models](#scatter-mapping-models))
- `fovCoords` - Coordinate system ('geo' tested, 'mag' supported)

### musicDataObj

Data object class for processed datasets ([music_data_object.py](pyDARNmusic/music/music_data_object.py)):
- Time series management
- Multi-dimensional data arrays (time × beam × range)
- Metadata handling
- Processing history tracking with timestamps
- Copy and transformation methods
- Active dataset management for processing pipeline

### Processing Functions

Core MUSIC algorithm steps available in [musicUtils.py](pyDARNmusic/utils/musicUtils.py):
- `defineLimits()` - Set time and range boundaries
- `applyLimits()` - Apply spatial/temporal limits
- `checkDataQuality()` - Validate data coverage
- `timeInterpolation()` - Uniform time sampling
- `beamInterpolation()` - Spatial interpolation across beams
- `filterTimes()` - Bandpass filtering for MSTID frequencies
- `detrend()` - Remove linear trends
- `windowData()` - Apply windowing function (Hanning)
- `calculateFFT()` - Fast Fourier Transform
- `calculateDlm()` - Cross-spectral matrix
- `calculateKarr()` - Horizontal wavenumber array (k-space)
- `detectSignals()` - Identify wave components using watershed algorithm
- `simulator()` - Generate synthetic MSTID data for testing

## Scatter Mapping Models

pyDARNmusic supports multiple field-of-view models via the `fovModel` parameter:

- **`GS`** - Ground Scatter model (Bristow et al. 1994) - **Default for MSTID studies**
- `IS` - Standard SuperDARN ionospheric scatter model
- `S` - Standard projection model
- `E1` - Chisham E-region 1/2-hop model
- `F1` - Chisham F-region 1/2-hop model
- `F3` - Chisham F-region 1.5-hop model
- `C` - Chisham projection model
- `None` - Use provided elevation/altitude values directly

The GS model is recommended for MSTID analysis as it properly accounts for ground scatter geometry.

## Dependencies

### Core Scientific
- `numpy` - Array operations and numerical computing
- `scipy` - Signal processing and scientific computing
- `matplotlib` - Visualization
- `cartopy` - Geospatial plotting
- `scikit-image` - Watershed algorithm for signal detection

### SuperDARN Ecosystem
- `pydarn` - SuperDARN data utilities and radar information
- `pydarnio` - SuperDARN file I/O (used internally by pydarn)
- `aacgmv2` - Magnetic coordinate transformations

### Geospatial
- `pygeos` - Geometric operations (note: being deprecated in favor of shapely 2.0)
- `geos` - GEOS library bindings
- `pycurl` - Data downloading

### Utilities
- `tqdm` - Progress bars for data loading

**Note**: `pygeos` is being deprecated in favor of `shapely >= 2.0`. Future versions may update this dependency.

## Relationship to DARNtids

pyDARNmusic is designed for **single-event analysis** of MSTIDs, providing detailed characterization of individual events (typically 2-hour windows). It enables researchers to:
- Examine specific MSTID events in detail
- Extract wave parameters (wavelength, period, velocity, azimuth)
- Generate publication-quality visualizations
- Test and validate MSTID detection methods

For **multi-event statistical studies and climatologies**, see [DARNtids](https://github.com/w2naf-academia/DARNtids), which:
- Uses pyDARNmusic as its core MSTID detection engine
- Automates processing across thousands of events
- Provides MSTID event classification
- Enables multi-radar network analysis
- Generates seasonal and statistical visualizations
- Manages event databases (MongoDB)
- Stores results in HDF5 format

**Example workflow:**
1. Use **pyDARNmusic** to analyze and understand specific MSTID events
2. Use **DARNtids** to process years of data and identify statistical patterns

The 2014 paper (pyDARNmusic) focused on a climatology of events from a single radar, while the 2016 paper (DARNtids) extended this to multi-radar source analysis.

## Algorithm References

The MUSIC implementation follows:

> Samson, J. C., R. A. Greenwald, J. M. Ruohoniemi, T. J. Hughes, and D. D. Wallis (1990), Magnetometer and radar observations of magnetohydrodynamic cavity modes in the Earth's magnetosphere, *Can. J. Phys.*, 68, 1265-1290.

> Bristow, W. A., R. A. Greenwald, and J. C. Samson (1994), Identification of high-latitude acoustic gravity wave sources using the Goose Bay HF Radar, *J. Geophys. Res.*, 99(A1), 319-331, doi:[10.1029/93JA01470](https://doi.org/10.1029/93JA01470).

The original MUSIC algorithm:

> Schmidt, R. O. (1986), Multiple emitter location and signal parameter estimation, *IEEE Trans. Antennas Propag.*, AP-34(3), 276-280.

## Examples and Tutorials

### Example Scripts

- [scripts/music_script.py](scripts/music_script.py) - Basic working example of MUSIC analysis

### Jupyter Notebooks

Located in [notebooks/](notebooks/):
- `MUSIC.ipynb` - Introduction to MUSIC analysis workflow
- `MUSIC-Simulate.ipynb` - Generate and analyze synthetic MSTID data
- `MUSIC-Simulate-BPK.ipynb` - Buckland Park radar simulation
- `MUSICWORKING-BPK.ipynb` - Real Buckland Park radar analysis example

### Test Data

Sample datasets are provided in [TestData/](TestData/) for testing and learning.

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

For bug reports and feature requests, please use [GitHub Issues](https://github.com/w2naf-academia/pyDARNmusic/issues).

## Citation

If you use pyDARNmusic in your research, please cite the appropriate references:

### Foundational Publication
```bibtex
@article{frissell2014climatology,
  author = {Frissell, N. A. and Baker, J. B. H. and Ruohoniemi, J. M. and Gerrard, A. J. and Miller, E. S. and Marini, J. P. and West, M. L. and Bristow, W. A.},
  title = {Climatology of medium-scale traveling ionospheric disturbances observed by the midlatitude Blackstone SuperDARN radar},
  journal = {Journal of Geophysical Research: Space Physics},
  volume = {119},
  year = {2014},
  doi = {10.1002/2014JA019870},
  note = {Original implementation and application of pyDARNmusic to Blackstone radar observations}
}
```

### Software Citation
```bibtex
@software{pydarnmusic,
  author = {Frissell, Nathaniel A. and Tholley, Francis Hassan},
  title = {pyDARNmusic: Multiple Signal Classification Algorithm for SuperDARN MSTID Detection},
  url = {https://github.com/w2naf-academia/pyDARNmusic},
  year = {2023},
  note = {Python 3 implementation}
}
```

### Python 3 Migration
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

### MUSIC Algorithm
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

### Example Multi-Radar Statistical Application
If using pyDARNmusic via DARNtids for multi-event analysis, also cite:
```bibtex
@article{frissell2016sources,
  author = {Frissell, N. A. and Baker, J. B. H. and Ruohoniemi, J. M. and Greenwald, R. A. and Gerrard, A. J. and Miller, E. S. and West, M. L.},
  title = {Sources and characteristics of medium-scale traveling ionospheric disturbances observed by high-frequency radars in the North American sector},
  journal = {Journal of Geophysical Research: Space Physics},
  volume = {121},
  year = {2016},
  doi = {10.1002/2015JA022168},
  note = {Multi-radar statistical analysis using DARNtids, which extends pyDARNmusic}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE](LICENSE) file for details.

## Authors and Contributors

### Principal Investigator
- **Nathaniel A. Frissell, PhD** - Original developer (2013) and project lead
  - Email: nathaniel.frissell@scranton.edu
  - Website: https://hamsci.org

### Major Contributors
- **Francis Hassan Tholley, MS** - Python 3 reimplementation and modernization (2022-2023)
  - Complete algorithm rewrite for Python 3.8+
  - Performance optimization with NumPy vectorization
  - Migration from DaViTpy to PyDARN ecosystem
  - Enhanced visualization with Cartopy

## Acknowledgments

### Funding Support

The original pyDARNmusic development (2013-2014) was supported by:
- NSF Grant ATM-0946900
- NSF Grant ATM-0849031
- NSF Grant ATM-0734241

The Python 3 migration (2022-2023) was supported by:
- NASA Grant 80NSSC21K0002

### Community and Infrastructure

- SuperDARN community for radar data and infrastructure
- PyDARN development team for data access libraries
- DaViTpy legacy project for original implementation foundation
- HamSCI (Ham Radio Science Citizen Initiative)
- University of Scranton Department of Computing Sciences
- Virginia Tech Bradley Department of Electrical and Computer Engineering
- Claude (Anthropic) for documentation assistance

## Support

- **Issues**: [GitHub Issues](https://github.com/w2naf-academia/pyDARNmusic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/w2naf-academia/pyDARNmusic/discussions)
- **Email**: nathaniel.frissell@scranton.edu

## Related Projects

- [DARNtids](https://github.com/w2naf-academia/DARNtids) - Multi-event MSTID analysis pipeline using pyDARNmusic
- [pydarn](https://github.com/SuperDARN/pydarn) - Python library for SuperDARN data
- [pydarnio](https://github.com/SuperDARN/pyDARNio) - SuperDARN file I/O
- [SuperDARN](http://vt.superdarn.org/) - Global HF radar network
- [DaViTpy](https://github.com/vtsuperdarn/davitpy) - Legacy SuperDARN toolkit (deprecated, archived)
