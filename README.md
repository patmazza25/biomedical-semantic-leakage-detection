# Biomedical‑Ontology‑Based‑Semantic‑Leakage‑Detection

---

# 🔹 macOS / Linux

## 1. Install Python 3.12.7

1. Download from **python.org → Python 3.12.7**:
   [https://www.python.org/downloads/release/python-3127/](https://www.python.org/downloads/release/python-3127/)

   • macOS: choose **macOS 64‑bit universal2 installer (.pkg)**.
   
   • Linux: download **Gzipped source tarball**.

3. Install it:

* **macOS**: run the `.pkg` installer.
* **Linux**:

```bash
tar -xvf Python-3.12.7.tgz
cd Python-3.12.7
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall   # installs python3.12 without overwriting system python
```

3. Verify installation:

```bash
python3.12 --version
# Should print: Python 3.12.7
```

## 2. Create a virtual environment (venv)

Use Python 3.12 (replace with your installed version if different):

```bash
python3.12 -m venv venv-3.12
source venv-3.12/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install dependencies and SciSpaCy models

```bash
pip install torch transformers datasets evaluate scikit-learn numpy scipy pandas tqdm uvicorn flask jinja2 python-dotenv waitress anthropic openai peft spacy scispacy sentence-transformers cython networkx matplotlib https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_craft_md-0.5.4.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
```

## 4. Install `nmslib`

```bash
pip install "nmslib @ git+https://github.com/nmslib/nmslib.git/#subdirectory=python_bindings"
```

---

# 🔸 Windows

## 1. Install Python 3.12.7

Download from **python.org → Python 3.12.7**:
[https://www.python.org/downloads/release/python-3127/](https://www.python.org/downloads/release/python-3127/)
Choose **Windows installer (64‑bit)** and check **Add python.exe to PATH** during install.

Verify installed Python versions:

```powershell
py --list
```

## 2. Create a virtual environment (venv)

Create and activate the venv:

```powershell
py -3.12 -m venv venv-3.12
.\nvenv-3.12\Scripts\Activate.ps1
```

If activation is blocked, run this once, then retry activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
```

Upgrade packaging tools:

```powershell
pip install --upgrade pip setuptools wheel
```

## 3. Install dependencies and SciSpaCy models

```powershell
pip install torch transformers datasets evaluate scikit-learn numpy scipy pandas tqdm uvicorn flask jinja2 python-dotenv waitress anthropic openai peft spacy scispacy sentence-transformers cython networkx matplotlib https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_craft_md-0.5.4.tar.gz https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
```

## 4. Install Visual Studio (required for `nmslib` build)

`nmslib` uses native extensions that need MSVC to compile. Install **Visual Studio 2022 Preview (Community)** from:
[https://visualstudio.microsoft.com/vs/preview/#download-preview](https://visualstudio.microsoft.com/vs/preview/#download-preview)

During installation, choose **C++ desktop development** and check only:

* MSVC v143
* Windows 11 SDK
* C++ CMake tools for Windows

## 5. Build and install `nmslib`

Open a Developer PowerShell environment for x64, then install:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
```

Change to your project folder:

```powershell
cd "C:\Users\<you>\path\to\Biomedical-Ontology-Based-Semantic-Leakage-Detection"
```

Install `nmslib`:

```powershell
pip install "nmslib @ git+https://github.com/nmslib/nmslib.git/#subdirectory=python_bindings"
```

---

# ▶️ Run the app

From the project root (with your venv active):

```bash
python main.py
```

Then open:
[http://127.0.0.1:5005](http://127.0.0.1:5005)

---

## 💡 Notes and Troubleshooting

* **SciSpaCy models** are installed via pip URLs above, so you do not need to run `python -m spacy download ...` separately.
* **Apple Silicon (M‑series)**: PyTorch installs a CPU build by default; you may enable Metal Performance Shaders (MPS) if available in your environment. If CUDA is mentioned as unavailable, that is expected on macOS.
* **Windows activation policy**: If PowerShell blocks venv activation, use the `Set-ExecutionPolicy` command shown above.
* **Building `nmslib`**: If compilation fails, re‑open a fresh **Developer PowerShell for VS 2022 x64**, confirm the selected C++ components are installed, and retry the install command.
* **Keeping wheels current**: Upgrading `pip setuptools wheel` before package installs prevents many build issues.
