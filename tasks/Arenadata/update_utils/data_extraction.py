import pandas as pd
import os
import glob

def load_client_dataframes(dir='telecom100k/telecom100k/telecom100k/psx', num_files=6):
    """
    Load a specified number of CSV files from the clients directory into pandas DataFrames.
    
    Parameters:
    -----------
    client_dir : str
        Path to the directory containing client CSV files
    num_files : int
        Number of files to load (default: 6)
    
    Returns:
    --------
    list
        List of pandas DataFrames loaded from CSV files
    """
    # Get list of all CSV files in the client directory
    file_paths = glob.glob(os.path.join(dir, '*.csv'))
    
    # Take the first num_files files
    selected_files = file_paths[:num_files]
    
    # Load each file into a DataFrame
    dataframes = []
    for file_path in selected_files:
        print(f"Loading file: {file_path}")
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    return dataframes

def data_preparation(files, subscribers, client_df, plan_df, psxattrs_df):
    """
    Prepare data for training by concatenating multiple session dataframes and merging with additional information.
    
    Parameters:
    -----------
    files : list
        List of CSV file paths containing session data with columns:
        IdSubscriber, Duration, UpTx, DownTx, StartSession, IdSession
    subscribers : pandas.DataFrame
        DataFrame with subscriber information containing columns:
        IdClient, Status, IdOnPSX (corresponds to IdSubscriber in session files)
    client_df : pandas.DataFrame
        DataFrame with client information containing columns:
        Id, IdPlan
    plan_df : pandas.DataFrame
        DataFrame with plan information containing columns:
        Id (corresponds to IdPlan in client_df), Enabled, Attrs
    psxattrs_df : pandas.DataFrame
        DataFrame with PSX attributes containing columns:
        DateFormat, TZ, TransmitUnits, Delimiter, Id (corresponds to IdPSX)
        
    Returns:
    --------
    pandas.DataFrame
        A unified DataFrame with columns from all sources
    """
    import pandas as pd
    
    # Read each CSV file into a dataframe and concatenate them
    dataframes = []
    for file in files:
        dataframes.append(file)
    
    combined_sessions = pd.concat(dataframes, ignore_index=True)
    
    # Merge with subscribers information
    result = combined_sessions.merge(
        subscribers,
        left_on='IdSubscriber',
        right_on='IdOnPSX',
        how='left'
    )
    
    # Merge with client information
    result = result.merge(
        client_df,
        left_on='IdClient',
        right_on='Id',
        how='left',
        suffixes=('', '_client')
    )
    
    # Merge with plan information
    result = result.merge(
        plan_df,
        left_on='IdPlan',
        right_on='Id',
        how='left',
        suffixes=('', '_plan')
    )
    
    # Merge with PSX attributes
    result = result.merge(
        psxattrs_df,
        left_on='IdPSX',
        right_on='Id',
        how='left',
        suffixes=('', '_psx')
    )
    
    # Clean up duplicate columns and rename if needed
    columns_to_drop = [col for col in result.columns if col.endswith(('_client', '_plan', '_psx'))]
    result = result.drop(columns=columns_to_drop)
    result = result.rename(columns={'Duartion': 'Duration'})
    # Select and reorder columns
    # Include all relevant columns from the merged dataframes
    final_columns = [
        'IdClient', 'Status', 'IdSubscriber', 'Duration', 
        'UpTx', 'DownTx', 'StartSession', 'EndSession', 'IdSession',
        'IdPlan', 'Enabled', 'Attrs',
        'DateFormat', 'TZ', 'TransmitUnits', 'Delimiter'
    ]
    
    # Return only the columns that exist in the result
    existing_columns = [col for col in final_columns if col in result.columns]
    return result[existing_columns]

def update_client_files(df, output_dir='clients'):
    """
    Group data by IdClient and update/create individual client files.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing client data with columns:
        IdClient, Status, IdSubscriber, Duration, UpTx, DownTx, StartSession, IdSession
    output_dir : str
        Directory where client files will be stored/updated
        
    Returns:
    --------
    dict
        Dictionary with statistics about updated and created files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counters
    stats = {'updated': 0, 'created': 0}
    
    # Group data by IdClient
    grouped = df.groupby('IdClient')
    
    for client_id, client_data in grouped:
        file_path = os.path.join(output_dir, f'client_{client_id}.csv')
        
        if os.path.exists(file_path):
            # If file exists, read it and append new data
            existing_data = pd.read_csv(file_path)
            
            # Concatenate existing and new data, drop duplicates based on IdSession
            updated_data = pd.concat([existing_data, client_data], ignore_index=True)
            updated_data = updated_data.drop_duplicates(subset=['IdSession'], keep='last')
            
            # Sort by StartSession to maintain chronological order
            updated_data = updated_data.sort_values('StartSession')
            
            # Save updated data
            updated_data.to_csv(file_path, index=False)
            stats['updated'] += 1
        else:
            # If file doesn't exist, create new file
            client_data.to_csv(file_path, index=False)
            stats['created'] += 1
    
    return stats

    

