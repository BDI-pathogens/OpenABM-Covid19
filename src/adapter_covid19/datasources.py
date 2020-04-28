import abc
import os
from typing import Tuple, Mapping, Any, Union, Optional, Sequence

import numpy as np
import pandas as pd

from adapter_covid19.enums import Region, Sector, Age, FinalUse, PrimaryInput

ALL_ENUMS = [Region, Sector, Age, FinalUse, PrimaryInput]


class Reader:
    def __init__(self, data_path: str):
        """
        Helper class to read data from disk

        Parameters
        ----------
        data_path: path to data
        """
        self.data_path = data_path

    def _get_filepath(self, filename: str) -> str:
        return os.path.join(self.data_path, filename)

    def load_csv(
        self,
        filename: str,
        orient: str = "dataframe",
        index_col: Optional[Union[int, Sequence[int]]] = None,
    ) -> Union[pd.DataFrame, Mapping[str, Any]]:
        data = pd.read_csv(self._get_filepath(f"{filename}.csv"), index_col=index_col)
        if orient.lower() == "dataframe":
            return data
        return data.to_dict(orient)


class DataSource(abc.ABC):
    def __init__(self, filename: str):
        """
        Read and parse a dataset from disk
        Parameters
        ----------
        filename: filename of dataset
        """
        self.filename = filename

    @abc.abstractmethod
    def load(self, reader: Reader) -> Any:
        """
        Load datasource from disk

        Parameters
        ----------
        reader: Reader class

        Returns
        -------
        Loaded and parsed data
        """
        raise NotImplementedError


class RegionDataSource(DataSource):
    def load(
        self, reader: Reader
    ) -> Union[Mapping[Region, float], Mapping[str, Mapping[Region, float]]]:
        data = reader.load_csv(self.filename, orient="dict", index_col=0)
        data = {k: {Region[kk]: vv for kk, vv in v.items()} for k, v in data.items()}
        if len(data) > 1:
            return data
        return next(iter(data.values()))


class SectorDataSource(DataSource):
    def load(
        self, reader: Reader
    ) -> Union[Mapping[Sector, float], Mapping[str, Mapping[Sector, float]]]:
        data = reader.load_csv(self.filename, orient="dict", index_col=0)
        data = {k: {Sector[kk]: vv for kk, vv in v.items()} for k, v in data.items()}
        if len(data) > 1:
            return data
        return next(iter(data.values()))


class RegionSectorAgeDataSource(DataSource):
    def load(
        self, reader: Reader
    ) -> Union[
        Mapping[Tuple[Region, Sector, Age], float],
        Mapping[str, Mapping[Tuple[Region, Sector, Age], float]],
    ]:
        data = reader.load_csv(self.filename, orient="dict", index_col=[0, 1, 2])
        data = {
            k: {(Region[kk[0]], Sector[kk[1]], Age[kk[2]]): vv for kk, vv in v.items()}
            for k, v in data.items()
        }
        if len(data) > 1:
            return data
        return next(iter(data.values()))


class DataFrameDataSource(DataSource):
    def load(self, reader: Reader) -> pd.DataFrame:
        frame = reader.load_csv(self.filename)
        frame = frame.set_index(frame.columns[0])
        for enum in ALL_ENUMS:
            try:
                frame = frame.rename(index=lambda x: enum[x])
            except KeyError:
                pass
            else:
                break
        for enum in ALL_ENUMS:
            try:
                frame = frame.rename(columns=lambda x: enum[x])
            except KeyError:
                pass
            else:
                break
        return frame


class WeightMatrix(DataSource):
    def load(self, reader: Reader) -> np.array:
        frame = reader.load_csv(self.filename).set_index("Sector")
        return frame.values
