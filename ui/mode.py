import os
import shutil

def install_custom_mode():
    # Get the installation path of streamlit-ace
    import streamlit_ace
    ace_path = os.path.dirname(streamlit_ace.__file__)
    mode_path = os.path.join(ace_path, 'frontend', 'build')
    
    # Source and destination paths
    src_path = os.path.join(os.getcwd(), 'ui', 'mode-custom.js')
    dst_path = os.path.join(mode_path, 'mode-custom.js')
    
    print(f"Installing custom mode from: {src_path}")
    print(f"Installing custom mode to: {dst_path}")
    # Check if source file exists
    if os.path.exists(dst_path):
        os.remove(dst_path)
    
    # Move the file
    shutil.copy(src_path, dst_path)
    print(f"Custom mode file installed to: {dst_path}")

if __name__ == "__main__":
    install_custom_mode()