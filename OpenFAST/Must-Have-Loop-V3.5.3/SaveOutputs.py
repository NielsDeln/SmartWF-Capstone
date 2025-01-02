import numpy as np
import subprocess
import shutil

# Configuration
openfast_executable = "openfast_x64.exe"
output_directory = "../../Must_Should_Dataset"  # Directory to store results

min_speed = 5
max_speed = 25
speed_step = 0.1
wind_speeds = np.arange(min_speed, max_speed+speed_step, speed_step)

min_std = 0.
max_std = 2.5
std_step = 0.25
stdevs = np.arange(min_std, max_std+std_step, std_step)

repetition = 0
# Loop to run simulations
for std in stdevs:
    for wind in wind_speeds:
        #Change simulation input
        shutil.copyfile("1_Configuration/Inflow_files/ramp_wind.dat", "1_Configuration/Inflow_files/temp_ramp_wind.dat")
        with open("1_Configuration/Inflow_files/temp_ramp_wind.dat", 'a') as wind_file:
            for t in np.arange(0, 660):
                speed = np.random.normal(wind, std)
                wind_file.write(f"\n{t}    ")
                wind_file.write(f"{speed}    ")
                for n in range(7):
                    wind_file.write("0    ")
        shutil.copyfile("1_Configuration/Inflow_files/temp_ramp_wind.dat", f"{output_directory}/Inputs/w{wind:1.4f}_s{std:2.2f}_{repetition}_ms_in.dat")

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
        
        shutil.copy("1_Configuration\IEA-22MW-RWT\IEA-22-280-RWT-Monopile\IEA-22-280-RWT-Monopile.out", f"{output_directory}/'Outputs/w{wind:1.f}_s{std:2.f}_{repetition}_ms_out.out")