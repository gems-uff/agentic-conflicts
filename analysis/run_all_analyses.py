import os
import subprocess
import sys

def main():
    scripts = [
        "rq1_structural.py",
        "rq2_strategies.py",
        "rq3_resolver.py"
    ]
    
    analysis_dir = os.path.dirname(__file__)
    
    for script in scripts:
        print(f"=====================================")
        print(f"Running {script}...")
        print(f"=====================================")
        
        script_path = os.path.join(analysis_dir, script)
        result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(analysis_dir))
        
        if result.returncode != 0:
            print(f"Error executing {script}")
            sys.exit(result.returncode)
            
    print("\nAll analyses completed successfully! Check the 'results/' directory for the outputs.")

if __name__ == "__main__":
    main()
