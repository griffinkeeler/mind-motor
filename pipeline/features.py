from mne.decoding import CSP
from epochs import run_epoch_extraction


def create_csp_object(n_components: int, log: bool, reg='ledoit_wolf'):
    """
    Creates a configured MNE CSP object.

    :param n_components: Number of spatial filters to extract.
    :param reg: Regularization method.
    :param log: Whether to apply log transformation to variances.

    :return: CSP: Configured CSP instance.
    """
    return CSP(n_components=n_components,
               reg=reg,
               log=log)


def transform_csp(x_epochs, y_epochs, csp_object):
    """
    Fits CSP and transforms the data.

    fit(): How two classes differ in EEG patterns.
    Finds the best spatial filters to maximize this difference.
    transform(): CSP applies these filters to each EEG trial
    Reduces each trial to log-variances of filtered signals

    :param x_epochs:
    :param y_epochs:
    :param csp_object:

    :return:
    """
    return csp_object.fit_transform(x_epochs, y_epochs)


def extract_csp_features(filepath, ch_type):
    csp = create_csp_object(n_components=4, log=True)

    x, y = run_epoch_extraction(filepath=filepath,
                                ch_type=ch_type)

    x_csp = transform_csp(x_epochs=x, y_epochs=y, csp_object=csp)

    return csp, x_csp


def main():
    extract_csp_features(filepath='/Users/griffinkeeler'
                                  '/PycharmProjects/mind-'
                                  'motor/data/aa.mat',
                         ch_type='eeg')


if __name__ == "__main__":
    main()
