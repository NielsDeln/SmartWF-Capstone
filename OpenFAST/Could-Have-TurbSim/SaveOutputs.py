import numpy as np
import subprocess
import shutil

# Configuration
openfast_executable = "openfast_x64.exe"
turbsim_executable = 'TurbSim_x64.exe'
output_directory = "../4_Results"  # Directory to store results
num_simulations = 10 # Number of simulations to run
min_speed = 5
max_speed = 25
speeds = np.linspace(min_speed, max_speed, num_simulations)

# Loop to run simulations
for i in speeds:
    #Change simulation input if Turbsim
    s = round(i, 4)
    shutil.copyfile("1_Configuration/Inflow_files/TurbSim_input.inp", "1_Configuration/Inflow_files/TurbSim_input_temp.inp")
    with open("1_Configuration/Inflow_files/TurbSim_input_temp.inp", 'r') as wind_file:
        lines = wind_file.readlines()

    delimiter = '    '
    parts = lines[39].strip().split(delimiter, maxsplit=2)

    value, param, description = parts
    # Replace the value for the matching parameter
    lines[39] = f"{i}{delimiter}{param}{delimiter}{description}\n"

    # Write the updated lines back to the file
    with open("1_Configuration/Inflow_files/TurbSim_input_temp.inp", "w") as file:
        file.writelines(lines)
    shutil.copyfile("1_Configuration/Inflow_files/TurbSim_input_temp.inp", f"4_Results/Input/WS_Sim_{s}.inp")

    turbsim_input = "1_Configuration/Inflow_files/TurbSim_input_temp.inp"
    # Run Turbsim
    try:
        subprocess.run(
            [turbsim_executable, turbsim_input], 
            check=True
        )
        print(f"TurbSim field {s} generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation {i}: {e}")
    shutil.copyfile("1_Configuration/Inflow_files/TurbSim_input_temp.bts", f"4_Results/Input/WS_Sim_{s}.bts")

    # Define input/output file paths of OpenFast
    simulation_input = "1_Configuration\IEA-22MW-RWT\IEA-22-280-RWT-Monopile\IEA-22-280-RWT-Monopile.fst"

    # Run OpenFAST
    try:
        subprocess.run(
            [openfast_executable, simulation_input],
            check=True
        )
        print(f"Simulation {i} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation {i}: {e}")
    
    shutil.copy("1_Configuration\IEA-22MW-RWT\IEA-22-280-RWT-Monopile\IEA-22-280-RWT-Monopile.out", f"4_Results/Output/Sim_WS_{s}.out")