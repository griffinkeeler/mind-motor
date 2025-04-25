import matplotlib.pyplot as plt
from load_data import create_mne_info, extract_metadata, extract_channel_names, load_mat_file
from mne import pick_channels, pick_info
from mne.viz import plot_topomap
from features import extract_csp_features


def get_clean_channel_names(invalid_channels, mne_info):
    """
    Filters out invalid EEG channels.

    :param invalid_channels:
    :param mne_info:
    :return:
    """
    return [ch for ch in mne_info.ch_names if ch not in invalid_channels]


def plot_csp_patterns(clean_patterns, clean_info):
    """
    Plots CSP spatial patterns as topomaps.
    :param clean_patterns:
    :param clean_info:
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        plot_topomap(
            data=clean_patterns[i],
            pos=clean_info,
            axes=axes[i],
            show=False,
            contours=0,
            names=clean_info.ch_names
        )
        axes[i].set_title(f"Filter {i + 1}", fontsize=10)

    plt.suptitle("CSP Spatial Patterns")
    plt.tight_layout()
    plt.show()


def visualize_csp():
    """Full CSP visualization pipeline."""
    data = load_mat_file(filepath='/Users/griffinkeeler/PycharmProjects/'
                                  'mind-motor/data/aa.mat')

    csp, _ = extract_csp_features(filepath='/Users/griffinkeeler'
                                           '/PycharmProjects/mind-'
                                           'motor/data/aa.mat',
                                  ch_type='eeg')

    bad_channels = ['FAF5', 'FAF1', 'FAF2', 'FAF6', 'FFC7', 'FFC8', 'CFC7',
                    'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'CCP7',
                    'CCP8', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6',
                    'PCP8', 'OPO1', 'OPO2']

    # Prepare channel info
    nfo = extract_metadata(data)
    channel_names = extract_channel_names(nfo)
    info = create_mne_info(channel_names=channel_names,
                           fs=100,
                           ch_type='eeg')
    info.set_montage('standard_1005', on_missing='ignore')

    # Clean channel info
    keep_channels = get_clean_channel_names(invalid_channels=bad_channels, mne_info=info)
    picks = pick_channels(info.ch_names, include=keep_channels)
    info_clean = pick_info(info, picks)
    patterns_clean = csp.patterns_[:, picks]

    # Visualize
    plot_csp_patterns(clean_patterns=patterns_clean, clean_info=info_clean)


def main():
    visualize_csp()


if __name__ == "__main__":
    main()
