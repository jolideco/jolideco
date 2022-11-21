import pytest
from numpy.testing import assert_allclose
from jolideco.models import NPredCalibration, NPredCalibrations


@pytest.mark.parametrize("format", ["yaml", "fits"])
def test_flux_components_io(format, tmpdir):
    calibrations = NPredCalibrations()

    calibrations["dataset-1"] = NPredCalibration(
        shift_x=0.1, shift_y=0.1, background_norm=0.9
    )
    calibrations["dataset-2"] = NPredCalibration(
        shift_x=-0.2, shift_y=0.23, background_norm=1.05
    )

    filename = tmpdir / f"test.{format}"

    calibrations.write(filename=filename, format=format)

    calibrations_new = NPredCalibrations.read(filename=filename, format=format)

    data = calibrations.to_dict()["dataset-1"]

    data_new = calibrations_new.to_dict()["dataset-1"]

    assert_allclose(data["shift_x"], data_new["shift_x"])
    assert_allclose(data["shift_y"], data_new["shift_y"])
    assert_allclose(data["background_norm"], data_new["background_norm"])

    data = calibrations.to_dict()["dataset-2"]
    data_new = calibrations_new.to_dict()["dataset-2"]
    assert_allclose(data["shift_x"], data_new["shift_x"])
    assert_allclose(data["shift_y"], data_new["shift_y"])
    assert_allclose(data["background_norm"], data_new["background_norm"])
