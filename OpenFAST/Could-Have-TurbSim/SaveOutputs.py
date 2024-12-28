import numpy as np
import subprocess
import shutil

# Configuration
openfast_executable = "openfast_x64.exe"
turbsim_executable = 'TurbSim_x64.exe'
output_directory = "../../Could_Dataset"  # Directory to store results

min_speed = 5
max_speed = 25
speed_step = 0.1
wind_speeds = np.arange(min_speed, max_speed+speed_step, speed_step)

repetition = 0
# Loop to run simulations
for wind in wind_speeds:
    #Change simulation input
    shutil.copyfile("1_Configuration/Inflow_files/TurbSim_input.inp", "1_Configuration/Inflow_files/TurbSim_input_temp.inp")
    with open("1_Configuration/Inflow_files/TurbSim_input_temp.inp", 'r') as wind_file:
        lines = wind_file.readlines()

    delimiter = '    '
    parts = lines[39].strip().split(delimiter, maxsplit=2)

    value, param, description = parts
    # Replace the value for the matching parameter
    lines[39] = f"{wind}{delimiter}{param}{delimiter}{description}\n"

    # Write the updated lines back to the file
    with open("1_Configuration/Inflow_files/TurbSim_input_temp.inp", "w") as file:
        file.writelines(lines)
    shutil.copyfile("1_Configuration/Inflow_files/TurbSim_input_temp.inp", f"{output_directory}/Inputs/w{wind:1.f}_s{0:2.f}_{repetition}_c_in.inp")

    turbsim_input = "1_Configuration/Inflow_files/TurbSim_input_temp.inp"
    # Run Turbsim
    try:
        subprocess.run(
            [turbsim_executable, turbsim_input], 
            check=True
        )
        print(f"TurbSim field {wind} generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation {wind}: {e}")
    shutil.copyfile("1_Configuration/Inflow_files/TurbSim_input_temp.bts", f"{output_directory}/Inputs/w{wind:1.f}_s{0:2.f}_{repetition}_c_in.bts")

    # Define input/output file paths
    simulation_input = "1_Configuration\IEA-22MW-RWT\IEA-22-280-RWT-Monopile\IEA-22-280-RWT-Monopile.fst"

    # Run OpenFAST
    try:
        subprocess.run(
            [openfast_executable, simulation_input],
            check=True
        )
        print(f"Simulation {wind} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation {wind}: {e}")
    
    shutil.copy("1_Configuration\IEA-22MW-RWT\IEA-22-280-RWT-Monopile\IEA-22-280-RWT-Monopile.out", f"{output_directory}/'Outputs/w{wind:1.f}_s{0:2.f}_{repetition}_c_out.out")