import xml.etree.ElementTree as ET
import os, sys
import pandas as pd
import traci
from sumolib import checkBinary
import numpy as np
import time
import xlsxwriter as xw
from scipy.stats.qmc import Sobol


def writing_vType_xml(path, num_vars, names, parameter):
    tree = ET.parse(path)
    root = tree.getroot()

    for vType in root.findall("vType"):
        if vType.get("id") == "passenger":
            vType.set("length", str(5))
            vType.set("speedFactor", str(1.0))
            vType.set("speedDev", str(0.1))
            vType.set("minGap", str(2.5))
            vType.set("accel", str(2.6))
            vType.set("decel", str(4.5))
            vType.set("tau", str(1.0))
            vType.set("delta", str(4))
            vType.set("emergencyDecel", str(9))
            vType.set("lcStrategic", str(1.0))
            vType.set("lcCooperative", str(1.0))
            vType.set("lcSpeedGain", str(1.0))
            vType.set("lcKeepRight", str(1.0))
            vType.set("lcOvertakeRight", str(0))
            vType.set("lcOpposite", str(1.0))
            vType.set("lcLookaheadLeft", str(2.0))
            vType.set("lcSpeedGainRight", str(0.1))
            vType.set("lcSpeedGainLookahead", str(0))
            vType.set("lcOvertakeDeltaSpeedFactor", str(0))
            vType.set("lcKeepRightAcceptanceTime", str(-1))
            vType.set("lcCooperativeSpeed", str(1.0))
            vType.set("lcAssertive", str(1.0))
            vType.set("jmCrossingGap", str(10))
            vType.set("jmIgnoreFoeProb", str(0))
            vType.set("jmIgnoreFoeSpeed", str(0))
            vType.set("jmIgnoreJunctionFoeProb", str(0))
            vType.set("jmSigmaMinor", str(0.5))
            vType.set("jmStoplineGap", str(1))
            vType.set("jmTimegapMinor", str(1))
            vType.set("impatience", str(0.0))
        if vType.get("id") == "truck":
            vType.set("length", str(7.1))
            vType.set("speedFactor", str(1.0))
            vType.set("speedDev", str(0.05))
            vType.set("minGap", str(2.5))
            vType.set("accel", str(1.3))
            vType.set("decel", str(4))
            vType.set("tau", str(1.0))
            vType.set("delta", str(4))
            vType.set("emergencyDecel", str(7))
            vType.set("lcStrategic", str(1.0))
            vType.set("lcCooperative", str(1.0))
            vType.set("lcSpeedGain", str(1.0))
            vType.set("lcKeepRight", str(1.0))
            vType.set("lcOvertakeRight", str(0))
            vType.set("lcOpposite", str(1.0))
            vType.set("lcLookaheadLeft", str(2.0))
            vType.set("lcSpeedGainRight", str(0.1))
            vType.set("lcSpeedGainLookahead", str(0))
            vType.set("lcOvertakeDeltaSpeedFactor", str(0))
            vType.set("lcKeepRightAcceptanceTime", str(-1))
            vType.set("lcCooperativeSpeed", str(1.0))
            vType.set("lcAssertive", str(1.0))
            vType.set("jmCrossingGap", str(10))
            vType.set("jmIgnoreFoeProb", str(0))
            vType.set("jmIgnoreFoeSpeed", str(0))
            vType.set("jmIgnoreJunctionFoeProb", str(0))
            vType.set("jmSigmaMinor", str(0.5))
            vType.set("jmStoplineGap", str(1))
            vType.set("jmTimegapMinor", str(1))
            vType.set("impatience", str(0.0))

    parameter_passenger = {}
    parameter_truck = {}
    for i in range(num_vars):
        if names[i][:3] == 'psg':
            parameter_passenger[names[i][4:]] = parameter[names[i]]
        else:
            parameter_truck[names[i][4:]] = parameter[names[i]]

    for vType in root.findall("vType"):
        if vType.get("id") == "passenger":
            for key, value in parameter_passenger.items():
                if key == 'lcKeepRightAcceptanceTime' and value < 0:
                    vType.set("lcKeepRightAcceptanceTime", str(-1))
                else:
                    vType.set(key, str(value))
        if vType.get("id") == "truck":
            for key, value in parameter_truck.items():
                if key == 'lcKeepRightAcceptanceTime' and value < 0:
                    vType.set("lcKeepRightAcceptanceTime", str(-1))
                else:
                    vType.set(key, str(value))

    tree.write(path, encoding="UTF-8", xml_declaration=True)


def run(duration):
    LaneArea_list = traci.lanearea.getIDList()
    LaneArea_veh_id = {}
    LaneArea_veh_num = {}
    for step in range(int(duration)):
        traci.simulationStep()
        for dec_id in LaneArea_list:
            zone_id = dec_id[:-2]
            veh_ids = list(traci.lanearea.getLastStepVehicleIDs(dec_id))
            if zone_id not in LaneArea_veh_id:
                LaneArea_veh_id[zone_id] = veh_ids
            else:
                LaneArea_veh_id[zone_id].extend(veh_ids)
        if step > 0 and step % 160 == 0:
            for zone_id in LaneArea_veh_id:
                if zone_id not in LaneArea_veh_num:
                    LaneArea_veh_num[zone_id] = [len(set(LaneArea_veh_id[zone_id]))]
                else:
                    LaneArea_veh_num[zone_id].append(len(set(LaneArea_veh_id[zone_id])))
                LaneArea_veh_id[zone_id] = []
    for zone_id in LaneArea_veh_id:
        LaneArea_veh_num[zone_id].append(len(set(LaneArea_veh_id[zone_id])))
    traci.close()
    return LaneArea_veh_num


def simulation_automatically(simulation_path, duration, seed):
    path = simulation_path + "\\test.sumocfg"
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    sumoBinary = checkBinary("sumo")

    traci.start([sumoBinary, "-c", path, "--seed", str(seed)])
    LaneArea_veh_num = run(duration)
    return LaneArea_veh_num


def run_simulation(num_vars, names, x, simulation_path, duration, seed):
    x = x.reshape(-1)
    # 首先修改参数
    vType_path = simulation_path + "\\basic.vType.xml"
    parameter = {}
    for i in range(num_vars):
        parameter[names[i]] = x[i]

    writing_vType_xml(vType_path, num_vars, names, parameter)
    bat_path = simulation_path
    os.chdir(bat_path)
    os.system("duarouter_run.bat")
    LaneArea_veh_num = simulation_automatically(simulation_path, duration, seed)
    return LaneArea_veh_num


def get_observation_flow(obs_path, start_time, end_time, obs_deta_time):
    path = obs_path + "/output.e1_out.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    Info = []
    duration = end_time - start_time
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        dec_id1 = interval.get("id")
        dec_id = dec_id1[:-2]
        veh = float(interval.get("nVehEntered"))
        Info.append([(t - start_time) // obs_deta_time, dec_id, veh])
    df_simulation = pd.DataFrame(Info)
    df_simulation.columns = ["interval", "id", "flow"]
    df_simulation = df_simulation.loc[df_simulation['interval'] < (duration // obs_deta_time)]
    return df_simulation


def get_observation_speed(obs_path, start_time, end_time, obs_deta_time):
    path = obs_path + "/output.e1_out.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    Info = []
    duration = end_time - start_time
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        dec_id1 = interval.get("id")
        dec_id = dec_id1[:-2]
        n = float(interval.get("nVehContrib"))
        speed = float(interval.get("speed"))
        Info.append([(t - start_time) // obs_deta_time, dec_id, speed, n])
    df_simulation = pd.DataFrame(Info)
    df_simulation.columns = ["interval", "id", "speed", "n"]
    df_simulation['speed'] = df_simulation['speed'] * df_simulation['n']
    df_simulation = df_simulation.groupby(["interval", "id"]).sum()
    df_simulation = df_simulation.reset_index()
    df_simulation['speed'] = df_simulation['speed'] / df_simulation['n']
    df_simulation = df_simulation.loc[df_simulation['interval'] < (duration // obs_deta_time)]
    return df_simulation


def get_observation_time(obs_path, start_time, end_time, obs_deta_time, LaneArea_veh_num):
    path = obs_path + "/output.e2_out.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    Info = []
    duration = end_time - start_time
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        dec_id1 = interval.get("id")
        dec_id = dec_id1[:-2]
        if float(interval.get("nVehSeen")) > 0:
            sampledSeconds = float(interval.get("sampledSeconds"))
            nVehSeen = float(interval.get("nVehSeen"))
            Info.append([(t - start_time) // obs_deta_time, dec_id, sampledSeconds, nVehSeen])
    df_simulation = pd.DataFrame(Info)
    df_simulation.columns = ["interval", "id", "sampledSeconds", "nVehSeen"]
    df_simulation = df_simulation.loc[df_simulation['interval'] < (duration // obs_deta_time)]
    df_simulation = df_simulation.groupby(["id", "interval"]).sum()
    df_simulation["travel_time"] = df_simulation["sampledSeconds"]
    df_simulation = df_simulation.reset_index()
    for row in df_simulation.itertuples():
        t_tot = getattr(row, 'travel_time')
        zone_id = getattr(row, 'id')
        interval = int(getattr(row, 'interval'))
        veh_num = LaneArea_veh_num[zone_id][interval]
        if veh_num > 0:
            df_simulation.at[row.Index, 'travel_time'] = t_tot / (veh_num)
    return df_simulation


def get_observation_queue(obs_path, start_time, end_time, obs_deta_time):
    path = obs_path + "/output.e2_out.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    Info = []
    duration = end_time - start_time
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        dec_id1 = interval.get("id")
        dec_id = dec_id1[:-2]
        queue_len = float(interval.get("meanMaxJamLengthInVehicles"))
        Info.append([(t - start_time) // obs_deta_time, dec_id, queue_len])
    df_simulation = pd.DataFrame(Info)
    df_simulation.columns = ["interval", "id", "mean_queue_num"]
    df_simulation = df_simulation.loc[df_simulation['interval'] < (duration // obs_deta_time)]
    df_simulation = df_simulation.groupby(["interval", "id"]).sum().reset_index()

    return df_simulation


def get_True_data(path):
    path_flow = path + "\\true_flow.csv"
    path_time = path + "\\true_time.csv"
    path_speed = path + "\\true_speed.csv"
    path_queue = path + "\\true_queue.csv"
    True_data_flow = pd.read_csv(path_flow)
    True_data_time = pd.read_csv(path_time)
    True_data_speed = pd.read_csv(path_speed)
    True_data_queue = pd.read_csv(path_queue)
    return True_data_flow, True_data_time, True_data_speed, True_data_queue


def ae(df, key_parameter):
    return abs(df[key_parameter + '_x'] - df[key_parameter + '_y'])


def cal_sim_mape(obs_df, true_df, key_id, key_interval, key_parameter):
    obs_data_group = obs_df.groupby([key_id, key_interval]).mean()[key_parameter]
    true_data_group = true_df.groupby([key_id, key_interval]).mean()[key_parameter]
    obs_true_df = pd.merge(obs_data_group, true_data_group, left_index=True, right_index=True)  # 横向合并

    obs_true_df['AE'] = obs_true_df.apply(ae, axis=1, args=(key_parameter,))
    WMAPE = obs_true_df['AE'].sum() / obs_true_df[key_parameter + '_y'].sum()

    return WMAPE


def objective_function(num_vars, names, x, simulation_path, obs_path, start_time, end_time, obs_delta_time,
                           true_flow,
                           true_time, true_speed, true_queue):
    duration = end_time - start_time
    seed = 42
    for i in range(10):
        LaneArea_veh_num = run_simulation(num_vars, names, x, simulation_path, duration, seed + i)
        obs_flow = get_observation_flow(obs_path, start_time, end_time, obs_delta_time)
        obs_time = get_observation_time(obs_path, start_time, end_time, obs_delta_time, LaneArea_veh_num)
        obs_speed = get_observation_speed(obs_path, start_time, end_time, obs_delta_time)
        obs_queue = get_observation_queue(obs_path, start_time, end_time, obs_delta_time)
        if i == 0:
            obs_flow_df = obs_flow.groupby(["id", "interval"]).mean()
            obs_time_df = obs_time.groupby(["id", "interval"]).mean()
            obs_speed_df = obs_speed.groupby(["id", "interval"]).mean()
            obs_queue_df = obs_queue.groupby(["id", "interval"]).mean()
        else:
            obs_flow_df = obs_flow_df._append(obs_flow.groupby(["id", "interval"]).mean())
            obs_time_df = obs_time_df._append(obs_time.groupby(["id", "interval"]).mean())
            obs_speed_df = obs_speed_df._append(obs_speed.groupby(["id", "interval"]).mean())
            obs_queue_df = obs_queue_df._append(obs_queue.groupby(["id", "interval"]).mean())

    WMAPE_flow = cal_sim_mape(obs_flow_df, true_flow, "id", "interval", "flow")
    WMAPE_time = cal_sim_mape(obs_time_df, true_time, "id", "interval", "travel_time")
    WMAPE_speed = cal_sim_mape(obs_speed_df, true_speed, "id", "interval", "speed")
    WMAPE_queue = cal_sim_mape(obs_queue_df, true_queue, "id", "interval", "mean_queue_num")
    y = [WMAPE_flow, WMAPE_speed, WMAPE_time, WMAPE_queue]
    y = np.array(y)
    return y


if __name__ == "__main__":
    def xw_toExcel(X, fileName):
        sample_size = X.shape[0]
        workbook = xw.Workbook(fileName)
        worksheet1 = workbook.add_worksheet("sheet1")
        worksheet1.activate()
        i = 1
        for j in range(sample_size):
            row = 'A' + str(i)
            worksheet1.write_row(row, X[j, :])
            i += 1
        workbook.close()


    num_vars_all = 60

    names_all = []
    names_all.append('psg_length')
    names_all.append('psg_speedFactor')
    names_all.append('psg_speedDev')
    names_all.append('psg_minGap')
    names_all.append('psg_accel')
    names_all.append('psg_decel')
    names_all.append('psg_tau')
    names_all.append('psg_delta')
    names_all.append('psg_emergencyDecel')
    names_all.append('psg_lcStrategic')
    names_all.append('psg_lcCooperative')
    names_all.append('psg_lcSpeedGain')
    names_all.append('psg_lcKeepRight')
    names_all.append('psg_lcOvertakeRight')
    names_all.append('psg_lcOpposite')
    names_all.append('psg_lcLookaheadLeft')
    names_all.append('psg_lcSpeedGainRight')
    names_all.append('psg_lcSpeedGainLookahead')
    names_all.append('psg_lcOvertakeDeltaSpeedFactor')
    names_all.append('psg_lcKeepRightAcceptanceTime')
    names_all.append('psg_lcCooperativeSpeed')
    names_all.append('psg_lcAssertive')
    names_all.append('psg_jmCrossingGap')
    names_all.append('psg_jmIgnoreFoeProb')
    names_all.append('psg_jmIgnoreFoeSpeed')
    names_all.append('psg_jmIgnoreJunctionFoeProb')
    names_all.append('psg_jmSigmaMinor')
    names_all.append('psg_jmStoplineGap')
    names_all.append('psg_jmTimegapMinor')
    names_all.append('psg_impatience')
    names_all.append('trk_length')
    names_all.append('trk_speedFactor')
    names_all.append('trk_speedDev')
    names_all.append('trk_minGap')
    names_all.append('trk_accel')
    names_all.append('trk_decel')
    names_all.append('trk_tau')
    names_all.append('trk_delta')
    names_all.append('trk_emergencyDecel')
    names_all.append('trk_lcStrategic')
    names_all.append('trk_lcCooperative')
    names_all.append('trk_lcSpeedGain')
    names_all.append('trk_lcKeepRight')
    names_all.append('trk_lcOvertakeRight')
    names_all.append('trk_lcOpposite')
    names_all.append('trk_lcLookaheadLeft')
    names_all.append('trk_lcSpeedGainRight')
    names_all.append('trk_lcSpeedGainLookahead')
    names_all.append('trk_lcOvertakeDeltaSpeedFactor')
    names_all.append('trk_lcKeepRightAcceptanceTime')
    names_all.append('trk_lcCooperativeSpeed')
    names_all.append('trk_lcAssertive')
    names_all.append('trk_jmCrossingGap')
    names_all.append('trk_jmIgnoreFoeProb')
    names_all.append('trk_jmIgnoreFoeSpeed')
    names_all.append('trk_jmIgnoreJunctionFoeProb')
    names_all.append('trk_jmSigmaMinor')
    names_all.append('trk_jmStoplineGap')
    names_all.append('trk_jmTimegapMinor')
    names_all.append('trk_impatience')

    bounds_all = [[3, 7],
                  [0.2, 2],
                  [0, 1],
                  [1, 10],
                  [1, 3],
                  [1, 6],
                  [1e-09, 3],
                  [0.1, 10],
                  [6, 10],
                  [0, 10],
                  [0, 1],
                  [0, 10],
                  [0, 10],
                  [0, 1],
                  [0, 10],
                  [0, 10],
                  [0, 10],
                  [0, 10],
                  [-1, 1],
                  [-1, 10],
                  [0, 1],
                  [1e-09, 10],
                  [0, 30],
                  [0, 1],
                  [0, 20 / 3.6],
                  [0, 1],
                  [0, 1],
                  [0, 6],
                  [0, 3],
                  [0, 10],
                  [7, 12],
                  [0.2, 2],
                  [0, 1],
                  [1, 10],
                  [1, 3],
                  [1, 6],
                  [1e-09, 3],
                  [0.1, 10],
                  [6, 10],
                  [0, 10],
                  [0, 1],
                  [0, 10],
                  [0, 10],
                  [0, 1],
                  [0, 10],
                  [0, 10],
                  [0, 10],
                  [0, 10],
                  [-1, 1],
                  [-1, 10],
                  [0, 1],
                  [1e-09, 10],
                  [0, 30],
                  [0, 1],
                  [0, 20 / 3.6],
                  [0, 1],
                  [0, 1],
                  [0, 6],
                  [0, 3],
                  [0, 10]]
    bounds_all = np.array(bounds_all)
    default_values = [5, 1.0, 0.1, 2.5, 2.6, 4.5, 1.0, 4, 9, 1.0, 1.0, 1.0, 1.0, 0, 1.0, 2.0, 0.1, 0, 0, -1, 1.0, 1.0,
                      10, 0, 0, 0, 0.5, 1, 1, 0.0, 7.1, 1.0, 0.05, 2.5, 1.3, 4, 1.0, 4, 7, 1.0, 1.0, 1.0, 1.0, 0, 1.0,
                      2.0, 0.1, 0, 0, -1, 1.0, 1.0, 10, 0, 0, 0, 0.5, 1, 1, 0.0]
    default_values = np.array(default_values)

    whether_to_optimize = np.ones(num_vars_all)
    num_vars = np.sum(whether_to_optimize == 1)
    names = [names_all[i] for i in np.where(whether_to_optimize == 1)[0]]
    bounds = bounds_all[np.where(whether_to_optimize == 1)[0]]

    simulation_path = os.getcwd()
    obs_path = os.getcwd()

    true_flow, true_time, true_speed, true_queue = get_True_data(obs_path)

    start_time = 0
    end_time = 900
    duration = end_time - start_time
    obs_delta_time = 160

    # 60个因素 考虑随机性
    sample_size = 600
    sobol_seq = Sobol(d=num_vars_all, seed=42)
    Training_set_60input_noisy = sobol_seq.random(sample_size)
    for i in range(num_vars_all):
        Training_set_60input_noisy[:, i] = bounds[i][0] + (
                bounds[i][1] - bounds[i][0]) * Training_set_60input_noisy[:, i]
    Training_set_60input_noisy[:, 19][Training_set_60input_noisy[:, 19] < 0] = -1  # lcKeepRightAcceptanceTime
    Training_set_60input_noisy[:, 49][Training_set_60input_noisy[:, 49] < 0] = -1  # lcKeepRightAcceptanceTime

    Training_set_60input_noisy = np.array(Training_set_60input_noisy)
    np.save('Training_set.npy', Training_set_60input_noisy)
    xw_toExcel(Training_set_60input_noisy, 'Training_set.xlsx')

    test_size = 1000
    Testing_set_60input_noisy = np.random.uniform(low=bounds_all[:, 0], high=bounds_all[:, 1],
                                                  size=(test_size, num_vars_all))
    Testing_set_60input_noisy[:, 19][Testing_set_60input_noisy[:, 19] < 0] = -1
    Testing_set_60input_noisy[:, 49][Testing_set_60input_noisy[:, 49] < 0] = -1

    Testing_set_60input_noisy = np.array(Testing_set_60input_noisy)
    np.save('Testing_set.npy', Testing_set_60input_noisy)
    xw_toExcel(Testing_set_60input_noisy, 'Testing_set.xlsx')

    Time = []
    obj = 4
    y_Training_set_60input_noisy = np.zeros((sample_size, obj))
    for i in range(sample_size):
        print('Training ' + str(i))
        time_to_start = time.time()
        y_Training_set_60input_noisy[i, :] = objective_function(num_vars, names, Training_set_60input_noisy[i, :],
                                                                    simulation_path, obs_path, start_time, end_time,
                                                                    obs_delta_time,
                                                                    true_flow,
                                                                    true_time, true_speed, true_queue)
        Time.append(time.time() - time_to_start)
        print(y_Training_set_60input_noisy[i, :])
    Time = np.array(Time)
    np.save('Time-Training-set.npy', Time)

    xw_toExcel(y_Training_set_60input_noisy, 'y_Training_set.xlsx')
    np.save('y_Training_set.npy', y_Training_set_60input_noisy)

    y_Testing_set_60input_noisy = np.zeros((test_size, obj))
    for i in range(test_size):
        print('Testing ' + str(i))
        y_Testing_set_60input_noisy[i, :] = objective_function(num_vars, names, Testing_set_60input_noisy[i, :],
                                                                   simulation_path, obs_path, start_time, end_time,
                                                                   obs_delta_time,
                                                                   true_flow,
                                                                   true_time, true_speed, true_queue)
    xw_toExcel(y_Testing_set_60input_noisy, 'y_Testing_set.xlsx')
    np.save('y_Testing.npy', y_Testing_set_60input_noisy)
