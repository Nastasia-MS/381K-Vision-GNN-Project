# Setup Instructions for Local Development

## Quick Start

### Option 1: Automated Setup (Recommended)
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate virtual environment:**
   ```bash
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Testing the Installation

After setup, test that everything works:
```bash
python test.py
```

## Running the Main Script

Once everything is installed, you can run the training script:
```bash
python dynamic_hypergraph_edge_attention.py
```

## Deactivating Virtual Environment

When you're done:
```bash
deactivate
```

## Notes

- Make sure you have Python 3.7 or higher installed
- For GPU support, you may need to install PyTorch with CUDA support:
  ```bash
  pip install torch torchvision torch-geometric --index-url https://download.pytorch.org/whl/cu118
  ```
  (Replace cu118 with your CUDA version)

