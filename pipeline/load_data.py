from scipy.io import loadmat
import numpy as np
import mne

def load_mat_file(filepath):
    """

    :param filepath:
    :return: dict_keys(['__header__', '__version__',
     '__globals__', 'mrk', 'nfo', 'cnt'])
    """
    return loadmat(filepath)

def extract_signal_data(data):
    """
    Extracts EEG signal data (samples x channels).
    :param data:
    :return:
    """
    return data['cnt']

def extract_events(data):
    """

    :param data:
    :return: MATLAB struct containing events in the EEG recording
    """
    return data['mrk'][0][0]

def extract_metadata(data):
    """

    :param data:
    :return: MATLAB struct containing metadata ('fs', 'clab')
    """
    return data['nfo'][0][0]

def extract_channel_names(nfo):
    """
    Extract channel names into a list.
    :param nfo:
    :return:
    """
    return [str(ch[0]) for ch in nfo['clab'][0]]

def extract_raw_signal(cnt):
    """
    Extracts the raw signal and transposes
    to (channels x samples) for MNE.
    :return:
    """
    return cnt.T

def extract_raw_labels(mrk):
    """
    Extracts raw labels from metadata.
    :param mrk:
    :return: Shape: (280,)
    """
    return mrk['y'][0]

def extract_event_positions(mrk):
    """
    Extracts event positions from metadata.
    :param mrk:
    :return:
    """
    return mrk['pos'][0]

def extract_sampling_rate(nfo):
    """
    Extracts the sampling rate (fs).
    :param nfo:
    :return:
    """
    return int(nfo['fs'][0][0])

def create_mask(raw_labels):
    """
    Creates a mask that is True when values are *not* nan.
    :param raw_labels:
    :return:
    """
    return ~np.isnan(raw_labels)

def keep_valid_labels(mask, raw_labels):
    """
    Applies the mask to keep only valid labels.
    :param mask:
    :param raw_labels:
    :return:
    """
    return raw_labels[mask]

def convert_valid_labels(valid_labels):
    """
    Converts values from 1 and 2 to 0 and 1.
    :param valid_labels:
    :return:
    """
    return valid_labels.astype(int) - 1

def clean_event_positions(event_positions, mask):
    """
    Filters event positions using
     a boolean mask to align with valid labels.
    :param event_positions:
    :param mask:
    :return:
    """
    return event_positions[mask]

def mne_create_info(channel_names, fs, ch_type):
    """
    Creates the MNE info object.
    :param channel_names:
    :param fs:
    :param ch_type:
    :return:
    """
    return mne.create_info(ch_names=channel_names,
                           sfreq=fs,
                           ch_types=ch_type)

def create_mne_raw(raw_signal, mne_info):
    """
    Creates the MNE RawArray.
    :param raw_signal:
    :param mne_info:
    :return:
    """
    return mne.io.RawArray(raw_signal, mne_info)

def create_mne_events(event_positions, final_labels):
    """
    Create the MNE events array
    :param event_positions:
    :param final_labels:
    :return: [sample_index, 0, event_id]
    """
    return np.column_stack((event_positions,
                            np.zeros(len(final_labels), dtype=int),
                            final_labels))

def load_subject_data(filepath: str, ch_type: str):
    data = load_mat_file(filepath)

    cnt = extract_signal_data(data)
    mrk = extract_events(data)
    nfo = extract_metadata(data)

    channel_names = extract_channel_names(nfo)

    x_raw = extract_raw_signal(cnt)
    labels_raw = extract_raw_labels(mrk)

    valid_positions = extract_event_positions(mrk)

    valid_mask = create_mask(labels_raw)

    clean_labels = keep_valid_labels(valid_mask, labels_raw)

    labels = convert_valid_labels(clean_labels)

    valid_positions = clean_event_positions(valid_positions,
                                            valid_mask)

    fs = extract_sampling_rate(nfo)

    info = mne_create_info(channel_names, fs, ch_type=ch_type)

    raw = create_mne_raw(raw_signal=x_raw, mne_info=info)

    events = create_mne_events(event_positions=valid_positions,
                               final_labels=labels)

    return raw, events

def main():
    load_subject_data(filepath='/Users/griffinkeeler/PycharmProjects/'
                               'mind-motor/data/aa.mat', ch_type='eeg')

if __name__ == "__main__":
    main()



