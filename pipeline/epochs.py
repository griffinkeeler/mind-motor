import mne
from load_data import load_subject_data

def extract_epochs(raw_object, event_array, event_id):
    """
    Extracts epochs from a Raw instance.
    :param raw_object:
    :param event_array:
    :param event_id:
    :return:
    """
    return mne.Epochs(
        raw=raw_object,
        events=event_array,
        event_id=event_id,
        tmin=0,
        tmax=2,
        baseline=None,
        preload=True
    )

def extract_epoch_data(extracted_epochs):
    """
    Extracts the data from each epoch.
    :param extracted_epochs:
    :return: (NumPy array): shape (n_epochs, n_channels, n_times)
    """
    return extracted_epochs.get_data()

def extract_epoch_targets(extracted_epochs):
    """
    Extracts the labels for each event.
    :param extracted_epochs:
    :return: shape(n_epochs,) with values 0 or 1
    """
    return extracted_epochs.events[:, -1]

def run_epoch_extraction(filepath: str, ch_type: str):
    raw, events = load_subject_data(filepath=filepath,
                                    ch_type=ch_type)

    # Define the event IDs
    event_id_dict = dict(left=0, right=1)

    epochs = extract_epochs(raw_object=raw,
                            event_array=events,
                            event_id=event_id_dict)


    x_epochs = extract_epoch_data(epochs)
    y_epochs = extract_epoch_targets(epochs)

    return x_epochs, y_epochs


def main():
    run_epoch_extraction(filepath='/Users/griffinkeeler'
                                  '/PycharmProjects/mind-'
                                  'motor/data/aa.mat',
                         ch_type='eeg')

if __name__ == "__main__":
    main()
