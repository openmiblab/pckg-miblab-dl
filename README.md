# miblab-dl

Python API for miblab's trained deep-learning models

## Installation in a working environment

If you are working in an existing environment with pytorch already installed, 
then all you need is this:

```bash
pip install miblab-dl
```

## Installation from scratch

If you start from scratch, first create a virtual environment, activate it 
and make sure you have the latest pip version. 

On Windows:

```bash
python -m venv myenv
myenv/Scripts/activate
python -m pip install --upgrade pip
```

On Mac or Linux:

```bash
python -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip
```

Now install pytorch in your environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the `miblab-dl` package:

```bash
pip install miblab-dl
```

## Usage

You can import `miblab-dl` functionality in a python script like this:

```python
import miblab_dl as dl
```

Then to compute fat and water maps from numpy arrays representing 
`in_phase` and `opposed_phase` Dixon images:

```python
fat, water = dl.fatwater(opposed_phase, in_phase)
```