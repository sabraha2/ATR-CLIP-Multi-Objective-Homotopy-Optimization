import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process the CLIP dataset.")
    parser.add_argument('dataset_path', type=str, help='The path to the dataset file.')
    parser.add_argument('--test_scenarios', type=str, nargs='+', default=[], 
                        help='A list of scenario codes to include in the test set.')
    parser.add_argument('--train_output', type=str, default='training_set.csv', 
                        help='Filename for the output training set CSV.')
    parser.add_argument('--test_output', type=str, default='test_set.csv', 
                        help='Filename for the output test set CSV.')
    parser.add_argument('--val_output', type=str, default='validation_set.csv', 
                        help='Filename for the output validation set CSV.')
    parser.add_argument('--val_size', type=float, default=0.25, 
                        help='Size of the validation set as a fraction.')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='The seed used by the random number generator.')
    return parser.parse_args()

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def split_data(df, test_scenarios):
    """
    Split the dataset into training and test sets based on scenario codes.
    """
    df['Set'] = df.apply(lambda row: 'Test' if str(row['Scenario Code']) in test_scenarios else 'Train', axis=1)
    train_df = df[df['Set'] == 'Train'].drop(columns=['Set'])
    test_df = df[df['Set'] == 'Test'].drop(columns=['Set'])
    return train_df, test_df

def save_datasets(train_df, test_df, val_df, train_file, test_file, val_file):
    """
    Save the training, validation, and test datasets to CSV files.
    """
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    if val_df is not None:
        val_df.to_csv(val_file, index=False)

def main():
    args = parse_arguments()
    
    df = load_data(args.dataset_path)
    train_df, test_df = split_data(df, args.test_scenarios)
    
    if args.val_size > 0:
        train_df, val_df = train_test_split(train_df, test_size=args.val_size, random_state=args.random_state)
        save_datasets(train_df, test_df, val_df, args.train_output, args.test_output, args.val_output)
    else:
        save_datasets(train_df, test_df, None, args.train_output, args.test_output, args.val_output)

if __name__ == '__main__':
    main()
