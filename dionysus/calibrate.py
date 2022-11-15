# from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from . import enums


def get_calibrator(val_true, val_pred, val_std, task: enums.TaskType):
    # use validation set to do an isotonic fit
    calibrator = IsotonicRegression(out_of_bounds="clip")
    # calibrator = LinearRegression()

    if task == enums.TaskType.regression:
        # calibrator = calibrator.fit(val_std.ravel(), (val_true - val_pred).ravel())
        calibration = LinearRegression().fit(val_std, val_true - val_pred)
    elif task == enums.TaskType.binary:
        calibrator = calibrator.fit(val_std.ravel(), val_true.ravel())
    else:
        raise ValueError('Task type not implemented for calibration.')

    return calibrator


def apply_calibration(y_pred, y_std, calibrator: IsotonicRegression, task: enums.TaskType):
    if task == enums.TaskType.regression:
        # TODO
        return
    elif task == enums.TaskType.binary:
        predicted_prob = calibrator.predict(y_std)

        # deal with probabilities exceeding [0,1]
        predicted_prob[(1.0 < predicted_prob) & (predicted_prob <= 1.0 + 1e-5)] = 1.0
        return predicted_prob
    # y_pred += predicted_error
    # y_std  predicted_error
