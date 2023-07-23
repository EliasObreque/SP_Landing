
import pandas as pd


class ThrustMeasure:
    def __init__(self, thruster_properties):
        self.thrust_by_file, self.time_profile = self.load_thrust_profile(
            thruster_properties['thrust_profile']['file_name'],
            thruster_properties['thrust_profile']['ThrustName'],
            thruster_properties['thrust_profile']['TimeName'])
        self.isp_by_file = thruster_properties['thrust_profile']['isp']
        self.dt_profile = thruster_properties['thrust_profile']['dt']
        self.current_burn_time = 0
        self.historical_mag_thrust = []
        self.t_ig = 0
        self.thr_is_on = False
        self.thr_is_burned = False
        self.current_time = 0
        self.current_mag_thrust_c = 0

    @staticmethod
    def load_thrust_profile(file_name, thrust_name, time_name):
        dataframe = pd.read_csv(file_name)
        return dataframe[thrust_name].values, dataframe[time_name].values

    def propagate(self):
        if self.thr_is_on:
            if self.current_burn_time == 0:
                self.current_mag_thrust_c = self.thrust_by_file[0]
                self.current_burn_time += self.dt_profile
            elif self.current_burn_time <= max(self.time_profile):
                self.current_mag_thrust_c = self.thrust_by_file[int(self.current_burn_time / self.dt_profile)]
                self.current_burn_time += self.dt_profile
            else:
                self.current_mag_thrust_c = 0
                self.thr_is_burned = True
                self.current_time += self.dt_profile
        else:
            self.current_mag_thrust_c = 0
            self.current_time += self.dt_profile

    def get_current_thrust(self):
        return self.current_mag_thrust_c