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


class Occupation(OrderedEnum):
    O11 = "11 Corporate Managers And Directors"
    O12 = "12 Other Managers And Proprietors"
    O21 = "21 Science, Engineering, Tech Professionals"
    O22 = "22 Health Professionals"
    O23 = "23 Teaching And Educational Professionals"
    O24 = "24 Business, Media And Public Service Professionals"
    O31 = "31 Science, Engineering ,Tech Associate Prof"
    O32 = "32 Health And Social Care Associate Professionals"
    O33 = "33 Protective Service Occupations"
    O34 = "34 Culture, Media And Sports Occupations"
    O35 = "35 Business, Public Service Associate Prof"
    O41 = "41 Administrative Occupations"
    O42 = "42 Secretarial And Related Occupations"
    O51 = "51 Skilled Agricultural And Related Trades"
    O52 = "52 Skilled Metal, Electrical, Electronic Trades"
    O53 = "53 Skilled Construction And Building Trades"
    O54 = "54 Textiles, Printing And Other Skilled Trades"
    O61 = "61 Caring Personal Service Occupations"
    O62 = "62 Leisure, Travel And Related Personal Servic"
    O71 = "71 Sales Occupations"
    O72 = "72 Customer Service Occupations"
    O81 = "81 Process, Plant And Machine Operatives"
    O82 = "82 Transport And Drivers And Operatives"
    O91 = "91 Elementary Trades And Related Occupations"
    O92 = "92 Elementary Administration And Service Occupations"


class LabourState(OrderedEnum):
    ill = enum.auto()
    wfh = enum.auto()
    working = enum.auto()
    furloughed = enum.auto()
    unemployed = enum.auto()
