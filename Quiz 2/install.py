import subprocess
import sys

def install():
    print("Installing base requirements...")
    # Install everything from requirements.txt which includes resampy>=0.4.3
    # We use --no-build-isolation as a safety net for any old dependencies that might creep in
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--no-build-isolation"])
    
    print("Installing basic-pitch without strict dependency checking...")
    # Install basic-pitch without dependencies because its setup.py explicitly restricts resampy to <0.4.3
    # which causes numpy build failures on Python 3.13.
    # Since its underlying dependencies (pretty_midi, mir_eval) are already installed via requirements.txt, this is perfectly safe.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "basic-pitch", "--no-deps"])
    
    print("\nInstallation complete! You can now run 'python app.py'")

if __name__ == "__main__":
    install()
