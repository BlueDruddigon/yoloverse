from typing import Any, Optional, Tuple

import yaml
from addict import Dict
from typing_extensions import Self


class ConfigDict(Dict):
    def __missing__(self, name: str) -> None:
        """this method is called when a key is not found in the dictionary

        :param name: (str) the name of the missing key
        :raises: (KeyError) if the key is not found in the dictionary
        """
        raise KeyError(name)

    def __getattr__(self, item: str) -> Any:
        """retrieve the value of the attribute specified by `item`

        :param item: (str) name of the attribute to retrieve
        :returns: (Any) the value of the attribute
        :raises:
            - (AttributeError) if the attribute is not found
            - (Exception) if any other exception occurs during the retrieval
        """
        try:  # try to retrieve the value of the attribute by using parent class's __getattr__ method
            value = super().__getattr__(item)
        except KeyError:  # if the attribute is not found
            ex = AttributeError(f'"{self.__class__.__name__}" object has no attribute "{item}"')
        except Exception as e:  # handle other exceptions
            ex = e
        else:  # if no exception occurs, return the value
            return value
        raise ex  # raise the exception


class Config:
    def __init__(
      self, cfg_dict: Optional[dict] = None, cfg_text: Optional[str] = None, fname: Optional[str] = None
    ) -> None:
        """initialize a Config object

        :param cfg_dict: (dict) a dictionary containing configuration options. default: None
        :param cfg_text: (str) a string containing configuration options. default: None
        :param fname: (str) the file name of a configuration file. default: None
        """
        if cfg_dict is None:  # handle invalid dictionary
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):  # type checking
            raise TypeError(f'cfg_dict must be a dict, but got {type(cfg_dict)}')

        # update object's attributes
        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super().__setattr__('_fname', fname)

        if cfg_text:  # if `cfg_text` is provided, assign it as text
            text = cfg_text
        elif fname:  # if `fname` is provided, read the file content as text
            with open(fname, 'r') as f:
                text = f.read()
        else:  # create empty string if both `cfg_text` and `fname` are not provided
            text = ''
        super().__setattr__('_text', text)  # update `text` attribute

    @property
    def fname(self) -> str:
        """ get the file name associated with the configuration """
        return self._fname

    @property
    def text(self) -> str:
        """ returns the text associated with this object """
        return self._text

    @staticmethod
    def fromfile(fname: str) -> Self:
        """load a configuration from a file

        :param fname: (str) the path to the configuration file
        :returns: (Config) the loaded cobfiguration object
        """
        cfg_dict, cfg_text = Config._file2dict(fname)  # load file content to dictionary
        return Config(cfg_dict, cfg_text=cfg_text, fname=fname)  # create a Config object from corresponding dict

    @staticmethod
    def _file2dict(fname: str) -> Tuple[dict, str]:
        """read a YAML file and convert it into a dictionary

        :param fname: (str) path to the YAML file
        :returns: a tuple containing the dictionary representation of the YAML file and the file content as a string
        :raises: (IOError) if the filetype is not supported (only .yaml files are supported)
        """
        fname = str(fname)  # ensure that the file name is a string
        if fname.endswith('.yaml'):  # check if the file is a YAML file
            with open(fname, mode='r') as f:
                cfg_dict = yaml.safe_load(f)  # safe load YAML content into a dictionary
        else:  # if it's not a YAML file
            raise IOError('Only .yaml file type are supported now!')
        cfg_text = fname + '\n'  # initialize a string starts with the file name
        with open(fname, mode='r') as f:
            cfg_text += f.read()  # add the content of the file to it

        return cfg_dict, cfg_text

    def __repr__(self) -> str:
        """ returns a string representation of the Config object """
        return f'Config (path: {self.fname}): {self._cfg_dict.__repr__()}'

    def __getattr__(self, name: str) -> Any:
        """retrieve the value of an attribute dynamically
        this method is called when an attribute is accessed that does not exist in the object
        it delegates the attribute lookup to the underlying `_cfg_dict` object

        :param name: (str) name of the attribute to retrieve
        :returns: (Any) the value of the attribute
        """
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """overrides the default `__setattr__` method to handle dictionary values

        :param name: (str) name of the attribute to set
        :param value: (Any) the value to assign to the attribute
        """
        if isinstance(value, dict):
            value = ConfigDict(value)  # if `value` is a dict, wrap it in a `ConfigDict` object
        self._cfg_dict.__setattr__(name, value)  # update it to the config dict
