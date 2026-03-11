# How to Run ReMindRAG

To run the application with a single command, you can use the provided reproduction script.

## Quick Start (Single Command)

If you are on Windows, run the following command in PowerShell from the root of the repository:

```powershell
.\reproduce.ps1
```

This script will:
1. Validate your environment (expects `venv` to exist or will help you set it up).
2. Verify API configuration in `.env`.
3. Execute the smoke test/demo located in `example/example.py`.

## Manual Execution

Alternatively, you can run the example directly:

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run example
cd example
python example.py
```

## Configuration

Ensure your `.env` file is populated with your `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
