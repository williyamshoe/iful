import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from iful.image_set import ImageSet
from iful.flat_modeling import FlatModel
from iful.simulation_api import SimulationMockImageSet, create_simulation_models

@pytest.fixture
def mock_imset(tmp_path):
    psf = np.ones((5, 5)) / 25.0
    psf_file = tmp_path / "psf.npy"
    np.save(psf_file, psf)

    imset = SimulationMockImageSet(
        size=10,
        pixscale_arcsec=0.05,
        zs=3.0,
        wavelengths_full=np.linspace(5000, 5200, 20),
        psf_path=str(psf_file)
    )
    imset.restwave_peaks = [1216.0]
    imset.init_spec_fit = [3.0, 10.0, 100.0, 1.0]
    return imset

def test_flat_model_initialization(mock_imset):
    fm = FlatModel(
        mock_imset,
        ["EPL_Q_PHI"],
        ["SERSIC_ELLIPSE"]
    )

    assert fm.lensmodel == ["EPL_Q_PHI"]
    assert fm.sourcemodel == ["SERSIC_ELLIPSE"]
    assert len(fm.multi_band_list) == 3

def test_iful_model_initialization(mock_imset):
    profiles = ["ARCTAN", "CONSTANT_FITTED_BH", "SERSIC"]
    fm, ifulmodel = create_simulation_models(
        mock_imset,
        theta_E=0.8,
        source_x=0.05,
        source_y=0.05,
        iful_profiles=profiles
    )

    assert ifulmodel.imset == mock_imset
    assert ifulmodel.iful_profiles == profiles
    assert ifulmodel.get_num_free_params() > 0

def test_iful_model_generate_residuals(mock_imset):
    profiles = ["ARCTAN", "CONSTANT_FITTED_BH", "SERSIC"]
    fm, ifulmodel = create_simulation_models(
        mock_imset,
        theta_E=0.8,
        source_x=0.05,
        source_y=0.05,
        iful_profiles=profiles
    )

    num_free = ifulmodel.get_num_free_params()
    assert num_free > 0

    params = np.zeros(num_free)
    params[0] = 0.8  # theta_E
    params[1] = 2.0  # gamma
    params[2] = 0.75 # q

    res = ifulmodel.generate_residuals(params)
    assert isinstance(res, (float, np.floating))
    assert not np.isnan(res)
