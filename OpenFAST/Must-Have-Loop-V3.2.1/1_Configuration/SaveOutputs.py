import numpy as np
import subprocess
import shutil

# Configuration
openfast_executable = "../openfast_x64.exe"
output_directory = "../Results"  # Directory to store results
num_simulations = 20  # Number of simulations to run
min_speed = 5 
max_speed = 25
speeds = np.arange(min_speed, max_speed, num_simulations)

# Loop to run simulations
for i in speeds:
    #Change simulation input
    shutil.copyfile("InputData/ramp_wind.dat", "InputData/temp_ramp_wind.dat")
    with open("InputData/temp_ramp_wind.dat", 'a') as wind_file:
        for t in np.arange(0, 600):
            speed = np.random.normal(i, 0.2)
            wind_file.write(f"\n{t}    ")
            wind_file.write(f"{speed}    ")
            for n in range(7):
                wind_file.write("0    ")
    shutil.copy("InputData/temp_ramp_wind.dat", f"../4_Results/Input/WS_Sim_{i}.dat")


    # Define input/output file paths
    simulation_input = "main.fst"

    # Run OpenFAST
    try:
        subprocess.run(
            [openfast_executable, simulation_input],
            check=True
        )
        print(f"Simulation {i} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation {i}: {e}")
    
    shutil.copy("main.out", f"../4_Results/Output/Sim_WS_{i}.out")