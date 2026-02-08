# PowerShell script to create a Python venv and install light requirements
# Usage: Open PowerShell in repo root and run: .\scripts\create_env.ps1

Write-Output "Creating virtual environment .venv..."
python -m venv .venv

Write-Output "Activating virtual environment..."
.\.venv\Scripts\Activate

Write-Output "Upgrading pip..."
python -m pip install --upgrade pip

Write-Output "Installing lightweight requirements from requirements.txt..."
pip install -r requirements.txt

Write-Output "If you want the FiftyOne desktop UI, run: pip install 'fiftyone[desktop]'"

Write-Output "Heavy frameworks (TensorFlow / PyTorch) are commented in requirements.txt.\nEdit this script to uncomment and install them when you're ready.\n"

Write-Output "Done. Activate the venv with: .\\.venv\\Scripts\\Activate"
