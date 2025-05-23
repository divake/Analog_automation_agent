# Simulation class given analog circuit parameters
#
# Author: Yue (Julien) Niu

import csv
import subprocess
import os


# Remove Docker command and use direct execution
# DOCKER_CMD = 'docker exec --user=asal rlinux8-2 /bin/tcsh -c'
# OCEAN_FILENAME = 'oceanScript.ocn'


class Simulator:
    
    def __init__(self, circuit_path, circuit_path_docker, circuit_params, params_path, ocean_filename):
        """
        :param circuit_path: circuit path in a host system
        :param circuit_path_docker: circuit path inside a docker container (not used anymore)
        :param circuit_params: defined circuit parameters
        :param params_path: circuit parameter value path
        """
        self.circuit_path = circuit_path
        self.circuit_def = circuit_path + '/' + ocean_filename
        self.circuit_params = circuit_params
        self.params_path = params_path
        
        # Remove Docker references
        # self.circuit_path_docker = circuit_path_docker
        # self.docker_cmd = DOCKER_CMD
        
        # store simulation results
        self.sim_results = []


    def run_sim(self):
        """Start a simulation
        Note that all simulation parameters are defined in input.scs file.
        If additional simulation functions need to be added, you should directly edit input.scs file.
        """
        # Use direct path to ocean executable that we found
        ocean_path = "/EDA_Tools/Cadence/IC618/tools/dfII/bin/ocean"
        
        # Check if the path exists, otherwise use the default command
        if not os.path.exists(ocean_path):
            print(f"[WARNING] Could not find Ocean at {ocean_path}, trying with module system")
            sim_cmd = f'cd {self.circuit_path} && module load cadence/virtuoso/6.18 && ocean -nograph -replay oceanScriptNew.ocn'
        else:
            # Always load the cadence module to avoid virtuoso not found errors
            sim_cmd = f'cd {self.circuit_path} && module load cadence/virtuoso/6.18 && {ocean_path} -nograph -replay oceanScriptNew.ocn'
            
        print(f"Running simulation: {sim_cmd}")
        
        # Use bash to ensure module command works properly
        ret = subprocess.call(sim_cmd, shell=True, executable='/bin/bash')
        
        if ret:
            print('[ERROR] cmd is not properly executed!!!')
            print(f'Command used: {sim_cmd}')
            print("Make sure Cadence environment is properly set up")

                
    def get_results(self):
        """Simply extract result from results.txt generated from ocean
        """
        cur_sim_result = {}
        result_path = self.circuit_path + '/results.txt'
        
        # Check if results file exists
        if not os.path.exists(result_path):
            print(f"[ERROR] Results file not found: {result_path}")
            return
            
        result_file = open(result_path, 'r')
        lines = result_file.readlines()
        result_file.close()  # Close the file properly
        
        for line in lines:
            if ':' in line:
                try:
                    parts = line.split(':')
                    metric = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    # Skip if value is empty
                    if not value_str:
                        print(f"[WARNING] Empty value for metric: {metric}")
                        continue
                        
                    value = float(value_str)
                    cur_sim_result[metric] = value
                    cur_sim_result['Error_'+metric] = 0
                    cur_sim_result[metric + '_GroundTruth'] = 0
                except ValueError as e:
                    print(f"[WARNING] Failed to parse value for line: {line.strip()} - Error: {e}")
                    continue  # Skip this line but continue processing
                except Exception as e:
                    print(f"[ERROR] Unexpected error parsing line: {line.strip()} - Error: {e}")
                    continue  # Skip this line but continue processing
                    
        # Only append if we have any results
        if cur_sim_result:
            self.sim_results.append(cur_sim_result)
        else:
            print("[WARNING] No metrics were parsed from the results file")


    def run_all(self, n=10, display=True):
        """Run all simulations by sweeping paramters defined in the .csv file
        """
        with open(self.params_path, mode='r') as param_file:
            param_dict = csv.DictReader(param_file)
            for i, line in enumerate(param_dict):
                for p in self.circuit_params:
                    if p not in line: continue
                    
                    self.circuit_params[p] = line[p]
            
                # edit circuit parameters in .scs file
                alter_circ_param(self.circuit_params, self.circuit_def)

                # start simulation
                self.run_sim()
                
                # get simulation results
                self.get_results()

                # calculate relative error
                self.calc_error(line)

                if display and i > 10 and (i+1) % 50 == 0:
                    print('{} points simulated.'.format(i+1))

                if n != -1 and i == n - 1: break


    def calc_error(self, perf_ref):
        """Calculate error compared to reference values
        :param perf_ref: reference performance
        """
        # If there are no simulation results, skip error calculation
        if not self.sim_results:
            print("[ERROR] No simulation results to calculate error.")
            return
            
        for key in self.sim_results[-1]:
            if 'Error' not in key and 'GroundTruth' not in key:  # only check actual values, not error
                val_actual = self.sim_results[-1][key]
                val_ref = float(perf_ref[key])
                val_ref_save = val_ref

                if 'VoltageGain' in key:
                    val_actual = 10 ** (val_actual / 20)
                    val_ref = 10 ** (val_ref / 20)
                    rel_error = abs(val_ref - val_actual) / abs(val_ref)
                elif 'ConversionGain' in key or 'PowerGain' in key or 'NoiseFigure' in key or 'S11' in key or 'S22' in key:
                    val_actual = 10 ** (val_actual / 10)
                    val_ref = 10 ** (val_ref / 10)
                    rel_error = abs(val_ref - val_actual) / abs(val_ref)
                else:
                    rel_error = abs(val_ref - val_actual) / abs(val_ref)

                self.sim_results[-1]['Error_' + key] = rel_error
                self.sim_results[-1][key + '_GroundTruth'] = val_ref_save


def alter_circ_param(new_params_values, ocean_path):
    scs_file = open(ocean_path, 'r')
    lines = scs_file.readlines()
    scs_file.close()  # Properly close the file
    
    # format for set variable values
    format_var = 'desVar(   \"{}\" {} )\n'

    # locate the line of circuit parameters
    for i, line in enumerate(lines):
        if 'desVar' in line:
            var = line.split('\"')[1]
            if var in new_params_values:
                if new_params_values[var] == 0: continue
                
                lines[i] = format_var.format(var, new_params_values[var])

    ocean_path_new = '/'.join(ocean_path.split('/')[0:-1]) + '/oceanScriptNew.ocn'
    ocean_file_new = open(ocean_path_new, 'w')
    ocean_file_new.writelines(lines)
    ocean_file_new.close()  # Properly close the file