from app.domain_profiles.base import DomainProfile
from app.domain_profiles.general import GeneralDomainProfile
from app.domain_profiles.legal import LegalDomainProfile
from app.domain_profiles.tax import TaxDomainProfile


_PROFILES: dict[str, DomainProfile] = {
    "general": GeneralDomainProfile(),
    "tax": TaxDomainProfile(),
    "legal": LegalDomainProfile(),
}


def get_domain_profile(name: str | None) -> DomainProfile:
    if not name:
        return _PROFILES["general"]
    return _PROFILES.get(str(name).strip().lower(), _PROFILES["general"])


def list_domain_profiles() -> list[str]:
    return sorted(_PROFILES.keys())
