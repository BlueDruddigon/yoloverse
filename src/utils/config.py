from typing import Any, Optional, Tuple

import yaml
from addict import Dict


class ConfigDict(Dict):
    def __missing__(self, name: str) -> None:
        raise KeyError(name)

    def __getattr__(self, item: str) -> Any:
        try:
            value = super().__getattr__(item)
        except KeyError:
            ex = AttributeError(f'"{self.__class__.__name__}" object has no attribute "{item}"')
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Config:
    def __init__(
      self, cfg_dict: Optional[dict] = None, cfg_text: Optional[str] = None, fname: Optional[str] = None
    ) -> None:
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f'cfg_dict must be a dict, but got {type(cfg_dict)}')

        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super().__setattr__('_fname', fname)

        if cfg_text:
            text = cfg_text
        elif fname:
            with open(fname, 'r') as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)

    @property
    def fname(self) -> str:
        return self._fname

    @property
    def text(self) -> str:
        return self._text

    @staticmethod
    def fromfile(fname: str):
        cfg_dict, cfg_text = Config._file2dict(fname)
        return Config(cfg_dict, cfg_text=cfg_text, fname=fname)

    @staticmethod
    def _file2dict(fname: str) -> Tuple[dict, str]:
        fname = str(fname)
        if fname.endswith('.yaml'):
            with open(fname, mode='r') as f:
                cfg_dict = yaml.safe_load(f)
        else:
            raise IOError('Only .yaml file type are supported now!')
        cfg_text = fname + '\n'
        with open(fname, mode='r') as f:
            cfg_text += f.read()

        return cfg_dict, cfg_text

    def __repr__(self) -> str:
        return f'Config (path: {self.fname}): {self._cfg_dict.__repr__()}'

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)
