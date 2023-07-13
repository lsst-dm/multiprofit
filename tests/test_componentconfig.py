from multiprofit.config import set_config_from_dict
from multiprofit.componentconfig import EllipticalComponentConfig


def test_EllipticalComponentConfig():
    config = EllipticalComponentConfig()
    config2 = EllipticalComponentConfig()
    set_config_from_dict(config2, config.toDict())
    assert config == config2
