import enum
from typing import Mapping


class OrderedEnum(enum.Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Sector(OrderedEnum):
    A_AGRICULTURE = "A: Agriculture, forestry and fishing"
    B_MINING = "B: Mining and quarrying"
    C_MANUFACTURING = "C: Manufacturing"
    D_ELECTRICITY = "D: Electricity, gas, steam and air conditioning supply"
    E_WATER = "E: Water supply; sewerage and waste management"
    F_CONSTRUCTION = "F: Construction"
    G_TRADE = "G: Wholesale and retail trade; repair of motor vehicles"
    H_TRANSPORT = "H: Transportation and storage"
    I_ACCOMODATION = "I: Accommodation and food service activities"
    J_COMMUNICATION = "J: Information and communication"
    K_FINANCIAL = "K: Financial and insurance activities"
    L_REAL_ESTATE = "L: Real estate activities"
    M_PROFESSIONAL = "M: Professional, scientific and technical activities"
    N_ADMINISTRATIVE = "N: Administrative and support service activities"
    O_PUBLIC = "O: Public administration and defence"
    P_EDUCATION = "P: Education"
    Q_HEALTH = "Q: Human health and social work activities"
    R_ARTS = "R: Arts, entertainment and recreation"
    S_OTHER = "S: Other service activities"
    T_HOUSEHOLD = "T: Activities of households"


class Region(OrderedEnum):
    C_NE = "UKC: North East"
    D_NW = "UKD: North West"
    E_YORKSHIRE = "UKE: Yorkshire and The Humber"
    F_EM = "UKF: East Midlands"
    G_WM = "UKG: West Midlands"
    H_E = "UKH: East of England"
    I_LONDON = "UKI: London"
    J_SE = "UKJ: South East"
    K_SW = "UKK: South West"
    L_WALES = "UKL: Wales"
    M_SCOTLAND = "UKM: Scotland"
    N_NI = "UKN: Northern Ireland"


class Age(OrderedEnum):
    # A0 = "15- = "16 - 17"
    A16 = "16 - 17"
    A18 = "18 - 24"
    A25 = "25 - 34"
    A35 = "35 - 49"
    A50 = "50 - 64"
    A65 = "65+"


class Age10Y(OrderedEnum):
    A0 = "population_0_9"
    A10 = "population_10_19"
    A20 = "population_20_29"
    A30 = "population_30_39"
    A40 = "population_40_49"
    A50 = "population_50_59"
    A60 = "population_60_69"
    A70 = "population_70_79"
    A80 = "population_80"


def age10y_to_age(age: Mapping[Age10Y, float]) -> Mapping[Age, float]:
    return {
        Age.A16: age[Age10Y.A10] / 5,
        Age.A18: age[Age10Y.A10] / 5 + age[Age10Y.A20] / 2,
        Age.A25: age[Age10Y.A20] / 2 + age[Age10Y.A30] / 2,
        Age.A35: age[Age10Y.A30] / 2 + age[Age10Y.A40],
        Age.A50: age[Age10Y.A50] + age[Age10Y.A60] / 2,
        Age.A65: age[Age10Y.A60] / 2 + age[Age10Y.A70] + age[Age10Y.A80],
    }


class LabourState(OrderedEnum):
    ILL = enum.auto()
    WFH = enum.auto()
    WORKING = enum.auto()
    FURLOUGHED = enum.auto()
    UNEMPLOYED = enum.auto()


class PrimaryInput(OrderedEnum):
    IMPORTS = "imports"
    TAXES_PRODUCTS = "taxes on products"
    TAXES_PRODUCTION = "taxes on production"
    COMPENSATION = "compensation"
    FIXED_CAPTIAL_CONSUMPTION = "fixed capital consumption"
    NET_OPERATING_SURPLUS = "net operating surplus"


class FinalUse(OrderedEnum):
    C = "consumption"
    K = "capital formation"
    E = "exports"


class M(OrderedEnum):
    I = "imports"
    L = "labour"
    K = "capital"
