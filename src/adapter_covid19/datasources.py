import abc
import os
from typing import Tuple, Mapping, Any, Union

import numpy as np
import pandas as pd

from adapter_covid19.enums import Region, Sector, Age


class Reader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def _get_filepath(self, filename: str) -> str:
        return os.path.join(self.data_path, filename)

    def load_csv(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self._get_filepath(f"{filename}.csv"))


class DataSource(abc.ABC):
    def __init__(self, filename: str):
        self.filename = filename

    @abc.abstractmethod
    def load(self, reader: Reader) -> Any:
        raise NotImplementedError


class RegionDataSource(DataSource):
    # TODO: the interface for all of these should be the same
    def load(
        self, reader: Reader
    ) -> Union[Mapping[Region, float], Mapping[Region, Mapping[str, float]]]:
        frame = reader.load_csv(self.filename)
        n_cols = frame.shape[1]
        if n_cols > 2:
            return {
                frame.columns[i]: {
                    Region[t.Region]: t[i] for t in frame.itertuples(index=False)
                }
                for i in range(1, n_cols)
            }
        return {Region[t.Region]: t[-1] for t in frame.itertuples(index=False)}


class SectorDataSource(DataSource):
    def load(self, reader: Reader) -> Mapping[Sector, float]:
        frame = reader.load_csv(self.filename)
        data = {Sector[t.Sector]: t[-1] for t in frame.itertuples(index=False)}
        return data


class RegionSectorAgeDataSource(DataSource):
    def load(self, reader: Reader) -> Mapping[Tuple[Region, Sector, Age], float]:
        frame = reader.load_csv(self.filename)
        data = {
            (Region[t.Region], Sector[t.Sector], Age[t.Age]): t[-1]
            for t in frame.itertuples(index=False)
        }
        return data


class WeightMatrix(DataSource):
    def load(self, reader: Reader) -> np.array:
        frame = reader.load_csv(self.filename).set_index("Sector")
        return frame.values
