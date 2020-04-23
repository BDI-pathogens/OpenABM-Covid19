import re
import string

# Utilities
UPPERCASE = set(string.ascii_uppercase)
URL_REGEX = re.compile(r'^(https?|file)://')

START_OF_TIME = 0
DAYS_IN_A_YEAR = 365.25
