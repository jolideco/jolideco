from jolideco.core import MAPDeconvolver


def test_map_deconvolver_str():
    deco = MAPDeconvolver(n_epochs=1_000)

    assert "n_epochs" in str(deco)
