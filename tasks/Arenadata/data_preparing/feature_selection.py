import pandas as pd

def final_feature_selection(df):
    df = df.drop(columns=['IdSubscriber', 'EthernetPlan', 'StartSessionUTC', 'SessionDuration'])
    return df

def selection_for_clusterisation(df):
    df = df[['AvgPackets', 'AvgUp', 'AvgDown', 'EndPoint']]
    return df
