import xml.etree.ElementTree as ET
import os, sys
import pandas as pd
import traci
from sumolib import checkBinary
import numpy as np
import sobol_seq
import time
import xlsxwriter as xw


def writing_vType_xml(path, num_vars, names, parameter):
    tree = ET.parse(path)
    root = tree.getroot()
    for vType in root.findall("vType"):
        if vType.get("id") == "car":
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
    for vType in root.findall("vType"):
        if vType.get("id") == "car":
            for i in range(num_vars):
                if names[i] == 'lcKeepRightAcceptanceTime' and parameter['lcKeepRightAcceptanceTime'] < 0:
                    vType.set("lcKeepRightAcceptanceTime", str(-1))
                else:
                    vType.set(names[i], str(parameter[names[i]]))

    tree.write(path, encoding="UTF-8", xml_declaration=True)


def run(duration, signal):
    step = 0
    n = 0
    for step in range(int(duration)):
        # print(step)
        if n < len(signal['startTime']):
            if step == signal['startTime'][n]:
                traci.trafficlight.setPhase('411', signal['index'][n])
                traci.trafficlight.setPhaseDuration('411', signal['eventDuration'][n])
                n += 1
        traci.simulationStep()
    traci.close()


def simulation_automatically(simulation_path, duration, signal, seed):
    path = simulation_path + "\\test.sumocfg"
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    sumoBinary = checkBinary("sumo")

    traci.start([sumoBinary, "-c", path, "--seed", str(seed), "--summary-output", f"{simulation_path}/summary.xml"])
    run(duration, signal)


def run_simulation(num_vars, names, x, simulation_path, signal, duration, seed):
    x = x.reshape(-1)
    vType_path = simulation_path + "\\basic.vType.xml"
    parameter = {}
    for i in range(num_vars):
        parameter[names[i]] = x[i]

    writing_vType_xml(vType_path, num_vars, names, parameter)
    bat_path = simulation_path
    os.chdir(bat_path)
    os.system("duarouter_run.bat")
    simulation_automatically(simulation_path, duration, signal, seed)


def get_observation_flow(obs_path, start_time, end_time, obs_delta_time, duration):
    path = obs_path + "\\output.e1_detector.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    Info = []

    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        dec_id1 = interval.get("id")
        dec_id = dec_id1.split('_')[-2]
        veh = float(interval.get("flow"))
        Info.append([(t - start_time) // obs_delta_time, dec_id, veh])
    df_simulation = pd.DataFrame(Info)
    df_simulation.columns = ["interval", "id", "flow"]
    df_simulation = df_simulation.loc[df_simulation['interval'] < (duration // obs_delta_time)]
    return df_simulation


def get_observation_speed(obs_path, start_time, end_time, obs_deta_time, duration):
    path = obs_path + "\\output.e1_detector.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    Info = []
    duration = end_time - start_time
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        dec_id1 = interval.get("id")
        dec_id = dec_id1.split('_')[-2]
        speed = float(interval.get("speed"))
        Info.append([(t - start_time) // obs_deta_time, dec_id, speed])
    df_simulation = pd.DataFrame(Info)
    df_simulation.columns = ["interval", "id", "speed"]
    df_simulation = df_simulation.loc[df_simulation['interval'] < (duration // obs_deta_time)]
    return df_simulation


def get_observation_time(obs_path, start_time, end_time, obs_deta_time, duration):
    path = obs_path + "\\output.e2_detector.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    Info = []
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        dec_id1 = interval.get("id")
        dec_id = dec_id1.split('_')[-2]
        if float(interval.get("nVehSeen")) > 0:
            sampledSeconds = float(interval.get("sampledSeconds"))
            nVehSeen = float(interval.get("nVehSeen"))
            Info.append([(t - start_time) // obs_deta_time, dec_id, sampledSeconds, nVehSeen])
    df_simulation = pd.DataFrame(Info)
    df_simulation.columns = ["interval", "id", "sampledSeconds", "nVehSeen"]
    df_simulation = df_simulation.loc[df_simulation['interval'] < (duration // obs_deta_time)]
    df_simulation = df_simulation.groupby(["id", "interval"]).sum()
    df_simulation["travel_time"] = df_simulation["sampledSeconds"] / df_simulation["nVehSeen"]
    df_simulation = df_simulation.reset_index()
    return df_simulation


def get_observation_queue(obs_path, start_time, end_time, obs_deta_time, duration):
    path = obs_path + "\\output.e2_detector.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    Info = []
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        dec_id1 = interval.get("id")
        dec_id = dec_id1.split('_')[-2]
        queue_len = float(interval.get("meanMaxJamLengthInVehicles"))
        Info.append([(t - start_time) // obs_deta_time, dec_id, queue_len])
    df_simulation = pd.DataFrame(Info)
    df_simulation.columns = ["interval", "id", "mean_queue_num"]
    df_simulation = df_simulation.loc[df_simulation['interval'] < (duration // obs_deta_time)]
    return df_simulation


def get_observation_timeLoss(obs_path, start_time, end_time, obs_deta_time, duration):
    path = obs_path + "\\output.e3_detector.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    Info = []
    for interval in root.findall("interval"):
        t = float(interval.get("begin"))
        dec_id = interval.get("id")
        timeLoss = float(interval.get("meanTimeLoss"))
        Info.append([(t - start_time) // obs_deta_time, dec_id, timeLoss])
    df_simulation = pd.DataFrame(Info)
    df_simulation.columns = ["interval", "id", "timeLoss"]
    df_simulation = df_simulation.loc[df_simulation['interval'] < (duration // obs_deta_time)]
    return df_simulation


def get_True_data(path):
    path_flow = path + "\\true_flow.csv"
    path_time = path + "\\true_time.csv"
    path_speed = path + "\\true_speed.csv"
    path_queue = path + "\\true_queue.csv"
    path_timeLoss = path + "\\true_timeLoss.csv"
    True_data_flow = pd.read_csv(path_flow)
    True_data_time = pd.read_csv(path_time)
    True_data_speed = pd.read_csv(path_speed)
    True_data_queue = pd.read_csv(path_queue)
    True_data_timeLoss = pd.read_csv(path_timeLoss)
    return True_data_flow, True_data_time, True_data_speed, True_data_queue, True_data_timeLoss


def ae(df, key_parameter):
    return abs(df[key_parameter + '_x'] - df[key_parameter + '_y'])


def cal_sim_mape(obs_df, true_df, key_id, key_interval, key_parameter):
    obs_data_group = obs_df.groupby([key_id, key_interval]).mean()[key_parameter]
    true_data_group = true_df.groupby([key_id, key_interval]).mean()[key_parameter]
    obs_true_df = pd.merge(obs_data_group, true_data_group, left_index=True, right_index=True)  # 横向合并

    obs_true_df['AE'] = obs_true_df.apply(ae, axis=1, args=(key_parameter,))
    WMAPE = obs_true_df['AE'].sum() / obs_true_df[key_parameter + '_y'].sum()

    return WMAPE


def objective_function(num_vars, names, x, simulation_path, obs_path, start_time, end_time, signal, obs_delta_time,
                           true_flow,
                           true_time, true_speed, true_queue, true_timeLoss, obj):
    duration = end_time - start_time
    y = np.zeros((10, obj))
    seed = 42
    for i in range(10):
        run_simulation(num_vars, names, x, simulation_path, signal, duration, seed + i)
        obs_flow_df = get_observation_flow(obs_path, start_time, end_time, obs_delta_time, duration)
        obs_time_df = get_observation_time(obs_path, start_time, end_time, obs_delta_time, duration)
        obs_speed_df = get_observation_speed(obs_path, start_time, end_time, obs_delta_time, duration)
        obs_queue_df = get_observation_queue(obs_path, start_time, end_time, obs_delta_time, duration)
        # obs_timeLoss_df = get_observation_timeLoss(obs_path, start_time, end_time, obs_delta_time, duration)
        MAPE_flow = cal_sim_mape(obs_flow_df, true_flow, "id", "interval", "flow")
        MAPE_time = cal_sim_mape(obs_time_df, true_time, "id", "interval", "travel_time")
        MAPE_speed = cal_sim_mape(obs_speed_df, true_speed, "id", "interval", "speed")
        MAPE_queue = cal_sim_mape(obs_queue_df, true_queue, "id", "interval", "mean_queue_num")
        y[i, 0] = MAPE_flow
        y[i, 1] = MAPE_speed
        y[i, 2] = MAPE_time
        y[i, 3] = MAPE_queue
        print(y[i, :])
    y = np.mean(y, axis=0)
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


    num_vars_all = 29

    names_all = []
    names_all.append('speedFactor')
    names_all.append('speedDev')
    names_all.append('minGap')
    names_all.append('accel')
    names_all.append('decel')
    names_all.append('tau')
    names_all.append('delta')
    names_all.append('emergencyDecel')
    names_all.append('lcStrategic')
    names_all.append('lcCooperative')
    names_all.append('lcSpeedGain')
    names_all.append('lcKeepRight')
    names_all.append('lcOvertakeRight')
    names_all.append('lcOpposite')
    names_all.append('lcLookaheadLeft')
    names_all.append('lcSpeedGainRight')
    names_all.append('lcSpeedGainLookahead')
    names_all.append('lcOvertakeDeltaSpeedFactor')
    names_all.append('lcKeepRightAcceptanceTime')
    names_all.append('lcCooperativeSpeed')
    names_all.append('lcAssertive')
    names_all.append('jmCrossingGap')
    names_all.append('jmIgnoreFoeProb')
    names_all.append('jmIgnoreFoeSpeed')
    names_all.append('jmIgnoreJunctionFoeProb')
    names_all.append('jmSigmaMinor')
    names_all.append('jmStoplineGap')
    names_all.append('jmTimegapMinor')
    names_all.append('impatience')

    bounds_all = [[0.2, 2],
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
    default_values = [1.0, 0.1, 2.5, 2.6, 4.5, 1.0, 4, 9, 1.0, 1.0, 1.0, 1.0, 0, 1.0, 2.0, 0.1, 0, 0, -1, 1.0, 1.0, 10,
                      0, 0, 0, 0.5, 1, 1, 0.0]
    default_values = np.array(default_values)

    whether_to_optimize = np.ones(num_vars_all)

    simulation_path = os.getcwd()
    obs_path = os.getcwd()

    true_flow, true_time, true_speed, true_queue, true_timeLoss = get_True_data(obs_path)
    signal_df = pd.read_csv('IntersectionA-Signal.csv')
    signal = {'startTime': list(signal_df['startTime']), 'eventDuration': list(signal_df['eventDuration']),
              'index': list(signal_df['index'])}

    start_time = 0
    end_time = 900
    duration = end_time - start_time
    obs_delta_time = 185

    # 29个因素 考虑随机性
    sample_size = 300
    Training_set_29input_noisy = sobol_seq.i4_sobol_generate(num_vars_all, sample_size)
    for i in range(num_vars_all):
        Training_set_29input_noisy[:, i] = bounds_all[i][0] + (
                bounds_all[i][1] - bounds_all[i][0]) * Training_set_29input_noisy[:, i]
    Training_set_29input_noisy[:, 18][Training_set_29input_noisy[:, 18] < 0] = -1  # lcKeepRightAcceptanceTime

    xw_toExcel(Training_set_29input_noisy, 'Training_set.xlsx')

    test_size = 1000
    Testing_set_29input_noisy = np.random.uniform(low=bounds_all[:, 0], high=bounds_all[:, 1],
                                                  size=(test_size, num_vars_all))
    Testing_set_29input_noisy[:, 18][Testing_set_29input_noisy[:, 18] < 0] = -1

    xw_toExcel(Testing_set_29input_noisy, 'Testing_set.xlsx')

    Time = []
    # 29个因素的输出 考虑随机性
    obj = 4
    y_Training_set_29input_noisy = np.zeros((sample_size, obj))
    for i in range(sample_size):
        print('Training '+ str(i))
        time_to_start = time.time()
        y_Training_set_29input_noisy[i, :] = objective_function(num_vars_all, names_all,
                                                                    Training_set_29input_noisy[i, :], simulation_path,
                                                                    obs_path,
                                                                    start_time, end_time,
                                                                    signal, obs_delta_time, true_flow,
                                                                    true_time, true_speed, true_queue, true_timeLoss, obj)
        Time.append(time.time() - time_to_start)
    Time = np.array(Time)
    np.save('Time-Training-set.npy', Time)

    xw_toExcel(y_Training_set_29input_noisy, 'y_Training_set.xlsx')

    y_Testing_set_29input_noisy = np.zeros((test_size, obj))
    for i in range(test_size):
        print('Testing '+ str(i))
        y_Testing_set_29input_noisy[i, :] = objective_function(num_vars_all, names_all,
                                                                   Testing_set_29input_noisy[i, :], simulation_path,
                                                                   obs_path,
                                                                   start_time, end_time,
                                                                   signal, obs_delta_time, true_flow,
                                                                   true_time, true_speed, true_queue, true_timeLoss, obj)
    xw_toExcel(y_Testing_set_29input_noisy, 'y_Testing_set.xlsx')



